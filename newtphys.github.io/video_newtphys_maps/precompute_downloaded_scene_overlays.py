#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops

FPS = 25
BLACK_THRESHOLD = 10
COLLISION_THRESHOLD = 8
INSTANCE_CHROMA_THRESHOLD = 28
MANIFEST_FILE = "manifest.json"
RENDER_FILE = "_fps-25_render.mp4"
OPTICAL_FLOW_FILE = "_fps-25_opticalflow.mp4"

MATERIAL_COLORS = {
    "plastic": (255, 102, 102),
    "metal": (114, 191, 255),
    "leather": (191, 127, 76),
    "wood": (222, 174, 110),
    "paper/cardboard": (255, 214, 112),
    "plush/fiberfill": (255, 105, 180),
    "ceramic": (186, 150, 255),
    "foam": (102, 255, 178),
    "fabric/textile": (0, 214, 255),
    "mixed (paper + plastic)": (255, 153, 204),
}
DEFAULT_MATERIAL_COLOR = (255, 255, 255)


@dataclass(frozen=True)
class OverlaySpec:
    label: str
    output_file: str
    source_dir: str | None
    processor: str
    group: str
    merge_mode: str = "alpha"
    additive_gain: float = 1.0


SCENE_LABEL_LOOKUP = {
    "dl3dv-random-3-seed-5-20251212-034543": "Random Trio",
    "dl3dv-random-5-seed-125-20251212-191943": "Random Quintet",
    "dl3dv-random-6-seed-174-20251213-072253": "Random Sextet A",
    "dl3dv-random-6-seed-181-20251213-073854": "Random Sextet B",
    "dl3dv-yms-variations-soft-4-seed-29-20251214-063652": "Soft Quartet",
    "dl3dv-yms-variations-soft-6-seed-14-20251214-160155": "Soft Sextet",
}


OVERLAY_SPECS = [
    OverlaySpec(
        label="Instances",
        output_file="_fps-25_instances.mp4",
        source_dir="instances",
        processor="instances",
        group="Segmentation",
    ),
    OverlaySpec(
        label="Materials",
        output_file="_fps-25_materials.mp4",
        source_dir="materials",
        processor="materials",
        group="Segmentation",
    ),
    OverlaySpec(
        label="Scene flow",
        output_file="_fps-25_sceneflow.mp4",
        source_dir="sceneflow",
        processor="sceneflow",
        group="Kinematics",
    ),
    OverlaySpec(
        label="Optical flow",
        output_file=OPTICAL_FLOW_FILE,
        source_dir="opticalflow",
        processor="opticalflow",
        group="Kinematics",
    ),
    OverlaySpec(
        label="Collision",
        output_file="_fps-25_force_collision.mp4",
        source_dir="force_collision",
        processor="collision",
        group="Physics",
        merge_mode="additive",
        additive_gain=4.0,
    ),
    OverlaySpec(
        label="Gravity",
        output_file="_fps-25_force_pt_wise-gravity.mp4",
        source_dir="force_pt_wise-gravity",
        processor="default",
        group="Physics",
        merge_mode="additive",
        additive_gain=1.25,
    ),
    OverlaySpec(
        label="Stress",
        output_file="_fps-25_force_defo_grad_wise-material.mp4",
        source_dir="force_defo_grad_wise-material",
        processor="stress",
        group="Physics",
        merge_mode="additive",
        additive_gain=3.0,
    ),
]


@dataclass
class SceneContext:
    source_dir: Path
    source_relpath: Path
    output_dir: Path
    simulation_path: Path
    simulation_data: dict
    render_frames: dict[int, Path]
    instance_frames: dict[int, Path]
    slug: str
    label: str
    has_soft_objects: bool
    instance_to_material_color: dict[tuple[int, int, int], tuple[int, int, int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build precomputed overlay videos from the downloaded image-sequence dataset."
    )
    parser.add_argument(
        "--downloaded-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "downloaded",
        help="Root folder containing downloaded simulation folders.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Destination root for generated scene folders and manifest.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Case-insensitive token used to filter scene relative paths or generated slugs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild videos even when outputs appear up to date.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable to use.",
    )
    return parser.parse_args()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def extract_seed(name: str) -> str | None:
    match = re.search(r"seed-(\d+)", name)
    return match.group(1) if match else None


def extract_timestamp(name: str) -> str | None:
    match = re.search(r"_(\d{8}_\d{6})$", name)
    return match.group(1) if match else None


def build_scene_slug(source_relpath: Path) -> str:
    parts = list(source_relpath.parts)
    leaf = parts[-1]
    prefix = parts[:-1]
    seed = extract_seed(leaf)
    timestamp = extract_timestamp(leaf)

    slug_parts = prefix[:]
    if seed:
        slug_parts.append(f"seed-{seed}")
    if timestamp:
        slug_parts.append(timestamp.replace("_", "-"))

    return slugify("-".join(slug_parts))


def build_scene_label(source_relpath: Path, slug: str) -> str:
    if slug in SCENE_LABEL_LOOKUP:
        return SCENE_LABEL_LOOKUP[slug]

    parts = list(source_relpath.parts)
    leaf = parts[-1]
    seed = extract_seed(leaf)

    if len(parts) >= 4 and parts[1] == "random":
        label = f"{parts[1]} {parts[2]}"
    elif len(parts) >= 5 and parts[1] == "yms-variations":
        label = f"{parts[2]} {parts[3]}"
    else:
        label = " / ".join(parts[1:-1]) or parts[0]

    if seed:
        return f"{label} / seed {seed}"
    return label


def list_frame_paths(frame_dir: Path) -> dict[int, Path]:
    frame_paths: dict[int, Path] = {}
    if not frame_dir.exists():
        return frame_paths

    for pattern in ("*.png", "*.jpg", "*.jpeg"):
        for path in frame_dir.glob(pattern):
            try:
                frame_paths[int(path.stem)] = path
            except ValueError:
                continue

    return dict(sorted(frame_paths.items()))


def needs_update(output_path: Path, source_paths: list[Path], force: bool) -> bool:
    if force or not output_path.exists():
        return True

    output_mtime = output_path.stat().st_mtime
    return any(source_path.stat().st_mtime > output_mtime for source_path in source_paths)


def list_scene_contexts(downloaded_root: Path, output_root: Path, selected_tokens: list[str]) -> list[SceneContext]:
    selected = [token.lower() for token in selected_tokens]
    contexts: list[SceneContext] = []

    for simulation_path in sorted(downloaded_root.glob("**/simulation.json")):
        source_dir = simulation_path.parent
        source_relpath = source_dir.relative_to(downloaded_root)
        slug = build_scene_slug(source_relpath)
        scene_key = f"{source_relpath.as_posix()} {slug}".lower()
        if selected and not all(token in scene_key for token in selected):
            continue

        simulation_data = json.loads(simulation_path.read_text())
        render_frames = list_frame_paths(source_dir / "render")
        if not render_frames:
            raise FileNotFoundError(f"Missing render frames under {source_dir / 'render'}")

        instance_frames = list_frame_paths(source_dir / "instances")
        contexts.append(
            SceneContext(
                source_dir=source_dir,
                source_relpath=source_relpath,
                output_dir=output_root / slug,
                simulation_path=simulation_path,
                simulation_data=simulation_data,
                render_frames=render_frames,
                instance_frames=instance_frames,
                slug=slug,
                label=build_scene_label(source_relpath, slug),
                has_soft_objects=scene_has_soft_objects(simulation_data),
                instance_to_material_color=build_material_color_map(simulation_data),
            )
        )

    return contexts


def scene_has_soft_objects(simulation_data: dict) -> bool:
    for obj_meta in simulation_data.get("objects", {}).values():
        sim_name = str(obj_meta.get("sim", "")).lower()
        if "soft" in sim_name or "softer" in sim_name:
            return True
    return False


def build_material_color_map(simulation_data: dict) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    classes = simulation_data.get("encoding", {}).get("classes", [])
    objects = simulation_data.get("objects", {})
    mapping: dict[tuple[int, int, int], tuple[int, int, int]] = {}

    for offset, object_id in enumerate(sorted(objects, key=lambda value: int(value)), start=2):
        if offset >= len(classes):
            break

        class_color = tuple(classes[offset])
        description = objects[object_id].get("description", {})
        material_group = description.get("material_group")
        material_color = MATERIAL_COLORS.get(material_group, DEFAULT_MATERIAL_COLOR)
        mapping[class_color] = material_color

    return mapping


def boost_saturation(image: Image.Image, factor: float) -> Image.Image:
    hsv = image.convert("HSV")
    hue, sat, value = hsv.split()
    sat = sat.point(lambda channel: min(255, int(channel * factor)))
    return Image.merge("HSV", (hue, sat, value)).convert("RGB")


def alpha_from_non_black(image: Image.Image) -> Image.Image:
    red, green, blue = image.split()
    brightness = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    mask = brightness.point(lambda channel: 255 if channel > BLACK_THRESHOLD else 0)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def alpha_from_threshold(image: Image.Image, threshold: int) -> Image.Image:
    red, green, blue = image.split()
    brightness = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    mask = brightness.point(lambda channel: 255 if channel > threshold else 0)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def collision_overlay(image: Image.Image, threshold: int) -> Image.Image:
    red, _, _ = image.split()
    alpha = red.point(
        lambda channel: 0
        if channel <= threshold
        else min(255, int((channel - threshold) * 255 / max(1, 255 - threshold)))
    )
    rgba = image.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def alpha_from_chroma(image: Image.Image) -> Image.Image:
    red, green, blue = image.split()
    max_rgb = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    min_rgb = ImageChops.darker(ImageChops.darker(red, green), blue)
    chroma = ImageChops.subtract(max_rgb, min_rgb)
    mask = chroma.point(lambda channel: 255 if channel > INSTANCE_CHROMA_THRESHOLD else 0)
    rgba = image.convert("RGBA")
    rgba.putalpha(mask)
    return rgba


def build_material_overlay(image: Image.Image, color_map: dict[tuple[int, int, int], tuple[int, int, int]]) -> Image.Image:
    rgb = image.convert("RGB")
    output = Image.new("RGBA", rgb.size, (0, 0, 0, 0))
    output.putdata(
        [
            (*color_map[pixel], 255) if pixel in color_map else (0, 0, 0, 0)
            for pixel in rgb.getdata()
        ]
    )
    return output


def process_overlay_frame(
    image_path: Path,
    processor: str,
    color_map: dict[tuple[int, int, int], tuple[int, int, int]],
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")

    if processor == "instances":
        return alpha_from_chroma(image)

    if processor == "materials":
        return build_material_overlay(image, color_map)

    if processor == "collision":
        return collision_overlay(image, COLLISION_THRESHOLD)

    if processor == "sceneflow":
        return alpha_from_non_black(boost_saturation(image, 2.0))

    if processor == "stress":
        return alpha_from_non_black(boost_saturation(image, 4.0))

    return alpha_from_non_black(image)


def masked_rgb_from_overlay(overlay: Image.Image) -> Image.Image:
    if overlay.mode != "RGBA":
        return overlay.convert("RGB")

    masked = Image.new("RGB", overlay.size, (0, 0, 0))
    masked.paste(overlay.convert("RGB"), mask=overlay.getchannel("A"))
    return masked


def scale_rgb_intensity(image: Image.Image, gain: float) -> Image.Image:
    if gain == 1.0:
        return image
    return image.point(lambda channel: min(255, int(channel * gain)))


def merge_overlay_frame(
    render_frame: Image.Image,
    overlay_frame: Image.Image,
    merge_mode: str,
    additive_gain: float,
) -> Image.Image:
    render_rgb = render_frame.convert("RGB")

    if merge_mode == "additive":
        additive_rgb = scale_rgb_intensity(masked_rgb_from_overlay(overlay_frame), additive_gain)
        return ImageChops.add(render_rgb, additive_rgb)

    return Image.alpha_composite(render_rgb.convert("RGBA"), overlay_frame.convert("RGBA")).convert("RGB")


def encode_png_sequence(ffmpeg_bin: str, frame_dir: Path, output_path: Path) -> None:
    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-framerate",
        str(FPS),
        "-i",
        str(frame_dir / "%06d.png"),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def stage_frame(image_path: Path, output_path: Path) -> None:
    Image.open(image_path).convert("RGB").save(output_path)


def build_render_video(context: SceneContext, ffmpeg_bin: str, force: bool, script_path: Path) -> None:
    output_path = context.output_dir / RENDER_FILE
    source_paths = [script_path, context.simulation_path, *context.render_frames.values()]
    if not needs_update(output_path, source_paths, force):
        print(f"skip  {output_path}")
        return

    context.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{context.slug}-render-", dir=context.output_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        for output_idx, frame_path in enumerate(context.render_frames.values()):
            stage_frame(frame_path, temp_dir / f"{output_idx:06d}.png")

        encode_png_sequence(ffmpeg_bin, temp_dir, output_path)

    print(f"build {output_path}")


def resolve_source_frames(
    context: SceneContext,
    spec: OverlaySpec,
) -> tuple[dict[int, Path], str | None, str]:
    if spec.processor == "opticalflow":
        return {}, (
            "Dense optical flow is not generated from downloaded/: camera poses are present, "
            "but there is no dense 3D scene flow/depth representation to project exactly."
        ), spec.processor

    if spec.processor == "stress" and not context.has_soft_objects:
        return {}, "Stress is only enabled for scenes that contain softer objects.", spec.processor

    if spec.processor == "materials":
        materials_dir = context.source_dir / "materials"
        materials_frames = list_frame_paths(materials_dir)
        if materials_frames:
            return materials_frames, None, "default"
        if context.instance_frames:
            return context.instance_frames, None, spec.processor
        return {}, "Materials require instance masks, but this scene has no instance frames.", spec.processor

    if not spec.source_dir:
        return {}, "No source directory configured for this overlay.", spec.processor

    source_dir = context.source_dir / spec.source_dir
    source_frames = list_frame_paths(source_dir)
    if source_frames:
        return source_frames, None, spec.processor

    return {}, f"No source frames found under {source_dir}.", spec.processor


def build_overlay_video(
    context: SceneContext,
    spec: OverlaySpec,
    ffmpeg_bin: str,
    force: bool,
    script_path: Path,
) -> tuple[bool, str | None]:
    source_frames, unavailable_reason, processor = resolve_source_frames(context, spec)
    if not source_frames:
        return False, unavailable_reason

    output_path = context.output_dir / spec.output_file
    source_paths = [script_path, context.simulation_path, *source_frames.values()]
    if not needs_update(output_path, source_paths, force):
        print(f"skip  {output_path}")
        return True, None

    context.output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{context.slug}-{spec.processor}-", dir=context.output_dir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        output_idx = 0

        for frame_idx, render_frame_path in context.render_frames.items():
            render_rgb = Image.open(render_frame_path).convert("RGB")
            source_frame_path = source_frames.get(frame_idx)

            if source_frame_path is None:
                composite = render_rgb
            else:
                overlay_rgba = process_overlay_frame(
                    image_path=source_frame_path,
                    processor=processor,
                    color_map=context.instance_to_material_color,
                )
                composite = merge_overlay_frame(
                    render_rgb,
                    overlay_rgba,
                    spec.merge_mode,
                    spec.additive_gain,
                )

            composite.save(temp_dir / f"{output_idx:06d}.png")
            output_idx += 1

        if output_idx == 0:
            return False, "No overlay frames lined up with render frames."

        encode_png_sequence(ffmpeg_bin, temp_dir, output_path)

    print(f"build {output_path}")
    return True, None


def build_manifest(
    output_root: Path,
    scene_entries: list[dict],
    script_path: Path,
    source_paths: list[Path],
    force: bool,
    selected_tokens: list[str],
) -> None:
    manifest_path = output_root / MANIFEST_FILE
    source_deps = [script_path, *source_paths]
    if not needs_update(manifest_path, source_deps, force=force):
        return

    merged_scene_entries = scene_entries
    if selected_tokens and manifest_path.exists():
        existing_payload = json.loads(manifest_path.read_text())
        preserved_entries = {
            entry["slug"]: entry
            for entry in existing_payload.get("scenes", [])
            if entry.get("slug") not in {scene["slug"] for scene in scene_entries}
        }
        for entry in scene_entries:
            preserved_entries[entry["slug"]] = entry
        merged_scene_entries = sorted(
            preserved_entries.values(),
            key=lambda entry: (entry.get("source_relpath", ""), entry.get("slug", "")),
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video_root": "./video_newtphys_maps_2",
        "overlays": [
            {
                "label": "RGB",
                "file": RENDER_FILE,
                "group": "RGB",
            },
            *[
                {
                    "label": spec.label,
                    "file": spec.output_file,
                    "group": spec.group,
                }
                for spec in OVERLAY_SPECS
            ],
        ],
        "scenes": merged_scene_entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"write {manifest_path}")


def main() -> int:
    args = parse_args()
    downloaded_root = args.downloaded_root.resolve()
    output_root = args.output_root.resolve()
    script_path = Path(__file__).resolve()

    if not downloaded_root.exists():
        raise FileNotFoundError(f"Downloaded root does not exist: {downloaded_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    scene_contexts = list_scene_contexts(
        downloaded_root=downloaded_root,
        output_root=output_root,
        selected_tokens=args.scene,
    )
    if not scene_contexts:
        selected_text = ", ".join(args.scene) if args.scene else "any scenes"
        raise FileNotFoundError(f"No downloaded scenes found for {selected_text} under {downloaded_root}")

    scene_entries: list[dict] = []
    manifest_source_paths: list[Path] = []

    for context in scene_contexts:
        print(f"scene {context.source_relpath.as_posix()}")
        build_render_video(context, ffmpeg_bin=args.ffmpeg_bin, force=args.force, script_path=script_path)

        available_files = [RENDER_FILE]
        disabled_files: dict[str, str] = {}
        frame_counts = {
            RENDER_FILE: len(context.render_frames),
        }

        for spec in OVERLAY_SPECS:
            available, reason = build_overlay_video(
                context=context,
                spec=spec,
                ffmpeg_bin=args.ffmpeg_bin,
                force=args.force,
                script_path=script_path,
            )
            if available:
                available_files.append(spec.output_file)
                output_file = context.output_dir / spec.output_file
                if output_file.exists():
                    frame_counts[spec.output_file] = int(
                        subprocess.check_output(
                            [
                                "ffprobe",
                                "-v",
                                "error",
                                "-count_frames",
                                "-select_streams",
                                "v:0",
                                "-show_entries",
                                "stream=nb_read_frames",
                                "-of",
                                "default=noprint_wrappers=1:nokey=1",
                                str(output_file),
                            ],
                            text=True,
                        ).strip()
                    )
            elif reason:
                disabled_files[spec.output_file] = reason

        scene_entries.append(
            {
                "label": context.label,
                "slug": context.slug,
                "path": f"./video_newtphys_maps_2/{context.slug}",
                "source_relpath": context.source_relpath.as_posix(),
                "has_soft_objects": context.has_soft_objects,
                "available_files": available_files,
                "disabled_files": disabled_files,
                "frame_counts": frame_counts,
            }
        )
        manifest_source_paths.extend([context.simulation_path, *context.output_dir.glob("*.mp4")])

    build_manifest(
        output_root=output_root,
        scene_entries=scene_entries,
        script_path=script_path,
        source_paths=manifest_source_paths,
        force=args.force,
        selected_tokens=args.scene,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"ffmpeg/ffprobe failed with exit code {error.returncode}", file=sys.stderr)
        raise
