import os
import glob

import imageio.v2 as imageio
import numpy as np

from PIL import Image
from tqdm import tqdm
from pathlib import Path

BLACK_THRESHOLD = 10
COLLISION_THRESHOLD = 8
RGB_ADD_WEIGHT = 0.35
COLLISION_ADD_GAIN = 6.0
GRAVITY_ADD_GAIN = 2.5
STRESS_ADD_GAIN = 8.0
STRESS_SATURATION_GAIN = 4.0

MAP_NAME = {
    "random/3/c-1_no-3_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-5_20251212_034543": "horse",
    "random/5/c-1_no-5_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-125_20251212_191943": "warehouse",
    "random/6/c-1_no-6_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-174_20251213_072253": "brawner hall",
    "random/6/c-1_no-6_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-181_20251213_073854": "tree",
}


def get_frames(simulations_json_file, folder_name):
    root_path = Path(simulations_json_file).parent
    frames_path = root_path / folder_name
    return sorted(frames_path.glob("*.png"))

def save_video(frames, new_path, video_name):
    new_path = Path(new_path)
    os.makedirs(new_path, exist_ok=True)
    new_file_name = new_path / f"{video_name}.mp4"

    print(f"Saving video to {new_file_name} with {len(frames)} frames.")

    with imageio.get_writer(new_file_name, fps=30) as writer:
        for frame in frames:
            writer.append_data(np.array(frame.convert("RGB")))


def overlay_instances(simulations_json_file, rgb_frames):
    
    instance_frames = get_frames(simulations_json_file, "instances")
    assert len(rgb_frames) == len(instance_frames), "Number of RGB frames and instance frames do not match."

    blended_frames = []
    for rgb_frame, instance_frame in tqdm(zip(rgb_frames, instance_frames), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")
        instance_rgba = np.array(Image.open(instance_frame).convert("RGBA"))

        # Remove background pixels from the instance image so only labeled regions overlay.
        background_mask = (
            np.all(instance_rgba[:, :, :3] == 255, axis=-1)
            | np.all(instance_rgba[:, :, :3] == 0, axis=-1)
        )
        instance_rgba[background_mask, 3] = 0

        instance_image = Image.fromarray(instance_rgba, mode="RGBA")
        blended_image = Image.alpha_composite(rgb_image, instance_image)

        blended_frames.append(blended_image)

    return blended_frames

def overlay_materials(simulations_json_file, rgb_frames):
    material_frames = get_frames(simulations_json_file, "materials")
    assert len(rgb_frames) == len(material_frames), "Number of RGB frames and material frames do not match."

    blended_frames = []
    for rgb_frame, material_frame in tqdm(zip(rgb_frames, material_frames), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")
        material_rgba = np.array(Image.open(material_frame).convert("RGBA"))

        # Remove background pixels from the material image so only labeled regions overlay.
        background_mask = (
            np.all(material_rgba[:, :, :3] == 255, axis=-1)
            | np.all(material_rgba[:, :, :3] == 0, axis=-1)
        )
        material_rgba[background_mask, 3] = 0

        material_image = Image.fromarray(material_rgba, mode="RGBA")
        blended_image = Image.alpha_composite(rgb_image, material_image)

        blended_frames.append(blended_image)

    return blended_frames

def overlay_scene_flow(simulations_json_file, rgb_frames):
    scene_flow_frames = get_frames(simulations_json_file, "sceneflow")
    assert len(rgb_frames) == len(scene_flow_frames), "Number of RGB frames and scene flow frames do not match."

    blended_frames = []
    for rgb_frame, scene_flow_frame in tqdm(zip(rgb_frames, scene_flow_frames), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")
        scene_flow_rgba = np.array(Image.open(scene_flow_frame).convert("RGBA"))

        # Remove background pixels from the scene flow image so only labeled regions overlay.
        background_mask = (
            np.all(scene_flow_rgba[:, :, :3] == 255, axis=-1)
            | np.all(scene_flow_rgba[:, :, :3] == 0, axis=-1)
        )
        scene_flow_rgba[background_mask, 3] = 0

        scene_flow_image = Image.fromarray(scene_flow_rgba, mode="RGBA")
        blended_image = Image.alpha_composite(rgb_image, scene_flow_image)

        blended_frames.append(blended_image)

    return blended_frames


def boost_saturation(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = Image.fromarray(image.astype(np.uint8), mode="RGB").convert("HSV")
    hsv_array = np.array(hsv, dtype=np.float32)
    hsv_array[:, :, 1] = np.clip(hsv_array[:, :, 1] * factor, 0, 255)
    return np.array(Image.fromarray(hsv_array.astype(np.uint8), mode="HSV").convert("RGB"), dtype=np.float32)


def non_black_alpha(image: np.ndarray, threshold: int = BLACK_THRESHOLD) -> np.ndarray:
    return (image.max(axis=-1) > threshold).astype(np.float32)


def collision_alpha(image: np.ndarray, threshold: int = COLLISION_THRESHOLD) -> np.ndarray:
    red = image[:, :, 0]
    return np.clip((red - threshold) / (255 - threshold), 0.0, 1.0).astype(np.float32)


def blend_additive(rgb_frame: Path, overlay: np.ndarray, alpha: np.ndarray, overlay_gain: float) -> Image.Image:
    rgb = np.array(Image.open(rgb_frame).convert("RGB"), dtype=np.float32)
    alpha = alpha[:, :, None]
    blended = rgb * (1.0 - alpha + RGB_ADD_WEIGHT * alpha) + overlay * overlay_gain * alpha
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")


def overlay_force(simulations_json_file: str, rgb_frames: list[Path], folder_name: str, overlay_gain: float, alpha_mode: str, saturation_gain: float = 1.0) -> list[Image.Image]:
    force_frames = get_frames(simulations_json_file, folder_name)
    assert len(rgb_frames) == len(force_frames), f"Number of RGB frames and {folder_name} frames do not match."

    blended_frames = []
    for rgb_frame, force_frame in tqdm(zip(rgb_frames, force_frames), total=len(rgb_frames)):
        overlay = np.array(Image.open(force_frame).convert("RGB"), dtype=np.float32)
        if saturation_gain != 1.0:
            overlay = boost_saturation(overlay, saturation_gain)
        alpha = collision_alpha(overlay) if alpha_mode == "collision" else non_black_alpha(overlay)
        blended_frames.append(blend_additive(rgb_frame, overlay, alpha, overlay_gain))

    return blended_frames


def overlay_collision(simulations_json_file: str, rgb_frames: list[Path]) -> list[Image.Image]:
    return overlay_force(simulations_json_file, rgb_frames, "force_collision", COLLISION_ADD_GAIN, "collision")


def overlay_gravity(simulations_json_file: str, rgb_frames: list[Path]) -> list[Image.Image]:
    return overlay_force(simulations_json_file, rgb_frames, "force_pt_wise-gravity", GRAVITY_ADD_GAIN, "default")


def overlay_stress(simulations_json_file: str, rgb_frames: list[Path]) -> list[Image.Image]:
    return overlay_force(
        simulations_json_file,
        rgb_frames,
        "force_defo_grad_wise-material",
        STRESS_ADD_GAIN,
        "default",
        saturation_gain=STRESS_SATURATION_GAIN,
    )



def compute_video(simulations_json_file, map_name, output_path):
    rgb_frames = get_frames(simulations_json_file, "render")

    ##########################################
    #                INSTANCES               #
    ##########################################

    print(f"Overlaying instance segmentation on RGB frames for {map_name}...")
    frames_instances = overlay_instances(simulations_json_file, rgb_frames)
    save_video(frames_instances, output_path / map_name, "instances")

    print(f"Overlaying instance segmentation on RGB frames for {map_name}...")
    frames_materials = overlay_materials(simulations_json_file, rgb_frames)
    save_video(frames_materials, output_path / map_name, "materials")

    ##########################################
    #                 FLOWS                  #
    ##########################################

    print(f"Overlaying scene flow on RGB frames for {map_name}...")
    frames_scene_flow = overlay_scene_flow(simulations_json_file, rgb_frames)
    save_video(frames_scene_flow, output_path / map_name, "scene_flow")

    # print(f"Overlaying optical flow on RGB frames for {map_name}...")
    # frames_scene_flow = overlay_optical_flow(simulations_json_file, rgb_frames)
    # save_video(frames_scene_flow, output_path / map_name, "/scene_flow")

    ##########################################
    #             COLLISION                  #
    ##########################################

    print(f"Overlaying collision on RGB frames for {map_name}...")
    frames_scene_flow = overlay_collision(simulations_json_file, rgb_frames)
    save_video(frames_scene_flow, output_path / map_name, "collision")

    print(f"Overlaying gravity on RGB frames for {map_name}...")
    frames_scene_flow = overlay_gravity(simulations_json_file, rgb_frames)
    save_video(frames_scene_flow, output_path / map_name, "gravity")

    print(f"Overlaying stress on RGB frames for {map_name}...")
    frames_scene_flow = overlay_stress(simulations_json_file, rgb_frames)
    save_video(frames_scene_flow, output_path / map_name, "stress")


def main():
    output_path = Path("/Users/sebastiancavada/Desktop/Compositional_physics_ECCV_26 (1)/newtphys.github.io/video_newtphys_maps_2/out")
    download_path = "/Users/sebastiancavada/Desktop/Compositional_physics_ECCV_26 (1)/newtphys.github.io/downloaded/dl3dv"

    # get all simulations.json files and path
    simulations_json_files = glob.glob(os.path.join(download_path, "**", "simulation.json"), recursive=True)

    print(f"Found {len(simulations_json_files)} simulation.json files.")

    for simulations_json_file in simulations_json_files[:1]:
        path_name = simulations_json_file.split("/dl3dv/")[1].replace("/simulation.json", "")
        map_name = MAP_NAME.get(path_name, "unknown")
        if map_name == "unknown":
            print(f"Warning: Map name for path '{path_name}' not found in MAP_NAME dictionary. Using 'unknown' as map name.")

        print(f"Processing {simulations_json_file} - {map_name}...")
        compute_video(simulations_json_file, map_name, output_path)

if __name__ == "__main__":
    main()
