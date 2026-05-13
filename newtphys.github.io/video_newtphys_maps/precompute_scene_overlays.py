#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

RENDER_FILE = "_fps-25_render.mp4"
SOURCE_SUFFIX = "_og"

BLACK_KEY_FILTER = (
    "[1:v]format=rgba,colorkey=black:0.04:0.08[fg];"
    "[0:v][fg]overlay=format=auto,format=yuv420p[v]"
)

SCENEFLOW_FILTER = (
    "[1:v]eq=saturation=2,format=rgba,colorkey=black:0.04:0.08[fg];"
    "[0:v][fg]overlay=format=auto,format=yuv420p[v]"
)

INSTANCES_FILTER = (
    "[1:v]format=rgba,"
    "geq="
    "r='r(X,Y)':"
    "g='g(X,Y)':"
    "b='b(X,Y)':"
    "a='if(gt((max(max(r(X,Y),g(X,Y)),b(X,Y))-min(min(r(X,Y),g(X,Y)),b(X,Y))),28),255,0)'"
    "[fg];"
    "[0:v][fg]overlay=format=auto,format=yuv420p[v]"
)

FILTERS_BY_FILENAME = {
    "_fps-25_sceneflow.mp4": SCENEFLOW_FILTER,
    "_fps-25_instances.mp4": INSTANCES_FILTER,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute RGB-plus-overlay videos from *_og scene folders."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root folder containing scene folders such as bench_og and shop_og.",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene name without the _og suffix. May be passed multiple times.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even when they appear up to date.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        default="ffmpeg",
        help="ffmpeg executable to use.",
    )
    return parser.parse_args()


def list_source_dirs(root: Path, selected_scenes: set[str]) -> list[Path]:
    source_dirs = []

    for path in sorted(root.iterdir()):
        if not path.is_dir() or not path.name.endswith(SOURCE_SUFFIX):
            continue

        scene_name = path.name[: -len(SOURCE_SUFFIX)]
        if selected_scenes and scene_name not in selected_scenes:
            continue

        source_dirs.append(path)

    return source_dirs


def destination_dir_for(source_dir: Path) -> Path:
    return source_dir.with_name(source_dir.name[: -len(SOURCE_SUFFIX)])


def needs_update(output_path: Path, source_paths: list[Path], force: bool) -> bool:
    if force or not output_path.exists():
        return True

    output_mtime = output_path.stat().st_mtime
    return any(source_path.stat().st_mtime > output_mtime for source_path in source_paths)


def copy_render(render_src: Path, render_dst: Path, force: bool) -> None:
    if not needs_update(render_dst, [render_src], force):
        print(f"skip  {render_dst}")
        return

    shutil.copy2(render_src, render_dst)
    print(f"copy  {render_dst}")


def build_overlay_video(
    ffmpeg_bin: str,
    render_src: Path,
    overlay_src: Path,
    output_path: Path,
    force: bool,
) -> None:
    if not needs_update(output_path, [render_src, overlay_src], force):
        print(f"skip  {output_path}")
        return

    filter_graph = FILTERS_BY_FILENAME.get(overlay_src.name, BLACK_KEY_FILTER)
    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-i",
        str(render_src),
        "-i",
        str(overlay_src),
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
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
    print(f"build {output_path}")


def build_scene(scene_source_dir: Path, ffmpeg_bin: str, force: bool) -> None:
    render_src = scene_source_dir / RENDER_FILE
    if not render_src.exists():
        raise FileNotFoundError(f"Missing render file: {render_src}")

    scene_output_dir = destination_dir_for(scene_source_dir)
    scene_output_dir.mkdir(parents=True, exist_ok=True)

    copy_render(render_src, scene_output_dir / RENDER_FILE, force)

    overlay_paths = sorted(
        path
        for path in scene_source_dir.glob("*.mp4")
        if path.name != RENDER_FILE
    )

    for overlay_src in overlay_paths:
        build_overlay_video(
            ffmpeg_bin=ffmpeg_bin,
            render_src=render_src,
            overlay_src=overlay_src,
            output_path=scene_output_dir / overlay_src.name,
            force=force,
        )


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    selected_scenes = set(args.scene)

    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    source_dirs = list_source_dirs(root, selected_scenes)
    if not source_dirs:
        scene_text = ", ".join(sorted(selected_scenes)) if selected_scenes else "any scenes"
        raise FileNotFoundError(f"No source folders found for {scene_text} under {root}")

    for source_dir in source_dirs:
        print(f"scene {source_dir.name}")
        build_scene(source_dir, ffmpeg_bin=args.ffmpeg_bin, force=args.force)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(f"ffmpeg failed with exit code {error.returncode}", file=sys.stderr)
        raise
