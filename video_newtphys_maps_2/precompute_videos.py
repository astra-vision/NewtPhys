import os
import glob
import json
import shutil
import imageio.v2 as imageio
import numpy as np

from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path

from scipy.spatial.transform import Rotation as R

from matplotlib.colors import rgb_to_hsv, hsv_to_rgb # Imported for fast HSV conversion

MAP_NAME = {
    "random/3/c-1_no-3_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-5_20251212_034543": "horse",
    "random/5/c-1_no-5_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-125_20251212_191943": "warehouse",
    "random/6/c-1_no-6_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-174_20251213_072253": "Brawner Hall",
    "random/6/c-1_no-6_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-181_20251213_073854": "tree",
    "yms-variations/soft/6/c-1_no-6_d-10_s-dl3dv-all_models-hf-gso_MLP-10_smooth_h-10-40_seed-14_20251214_160155": "office",
}
MAPPING_GSO_PATH = "./gso_mapping.json"
MAPPING_GSO_DATA = json.load(open(MAPPING_GSO_PATH))
INSTANCE_COLOR_COUNT = 128
INSTANCE_COLOR_SATURATION = 0.75
INSTANCE_COLOR_VALUE = 1.0
INSTANCE_COLOR_GOLDEN_ANGLE = 137.50776405003785
OPTICAL_FLOW_BACKGROUND_THRESHOLD = 20
AMODAL_MATERIAL_VISIBLE_ALPHA = 128
AMODAL_MATERIAL_HIDDEN_ALPHA = 64
GRAVITY_MASS_MIN_KG = 0.1
GRAVITY_MASS_MAX_KG = 2.0
GRAVITY_MIN_VISIBLE_INTENSITY = 0.18
GRAVITY_INTENSITY_GAMMA = 0.55
GRAVITY_RED = np.array([255.0, 0.0, 0.0], dtype=np.float32)
GRAVITY_LEGEND_MARGIN = 24
GRAVITY_LEGEND_WIDTH = 72
GRAVITY_LEGEND_BAR_WIDTH = 18
GRAVITY_LEGEND_HEIGHT = 150
GSO_OBJECT_IDS = tuple(MAPPING_GSO_DATA.keys())
GSO_OBJECT_NAME_TO_INDEX = {
    value["name"]: idx
    for idx, value in enumerate(MAPPING_GSO_DATA.values(), start=1)
}
GSO_OBJECT_ID_TO_INDEX = {
    object_id: idx
    for idx, object_id in enumerate(GSO_OBJECT_IDS, start=1)
}

# UTILS
def compute_extrinsic_matrix(eye, at, up=np.array([0, 1, 0])):
    """
    Computes the 3x4 World-to-Camera extrinsic matrix [R | t]
    using standard OpenCV conventions (+Z forward, +X right, +Y down).
    """
    # 1. Calculate Forward Vector (+Z)
    forward = at - eye
    forward = forward / np.linalg.norm(forward)
    
    # 2. Calculate Right Vector (+X)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6: # Handle edge case looking straight up/down
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    
    # 3. Calculate True Up Vector (+Y Down for OpenCV, flip sign if Y should be up)
    # Note: Standard CV images have the Y axis pointing DOWN. 
    true_down = np.cross(right, forward) 
    
    # 4. Construct Rotation Matrix (World to Camera)
    R = np.vstack([right, true_down, forward])
    
    # 5. Construct Translation Vector
    t = -R @ eye
    
    # Combine into 3x4 matrix
    return np.hstack([R, t.reshape(3, 1)])

def get_frames(simulations_json_file, folder_name):
    root_path = Path(simulations_json_file).parent
    frames_path = root_path / folder_name
    return sorted(frames_path.glob("*.png"))

def get_all_instances_frames(simulations_json_file):
    root_path = Path(simulations_json_file).parent
    instances_frames_path = root_path / "instances"
    all_images = sorted(instances_frames_path.glob("obj_*/*.png"))

    num_folders = len({img.parent for img in all_images})
    images_per_folder = len(all_images) // num_folders

    return all_images[images_per_folder:], num_folders - 1 # because object 0 is the background and we want to skip it 

def get_instance_index(object_id: int | str) -> int:
    if isinstance(object_id, int) or str(object_id).isdigit():
        class_id = int(object_id)
        assert 1 <= class_id <= INSTANCE_COLOR_COUNT, f"Class ID {class_id} must be in [1, {INSTANCE_COLOR_COUNT}]."
        return class_id

    object_id = str(object_id)
    if object_id in GSO_OBJECT_ID_TO_INDEX:
        return GSO_OBJECT_ID_TO_INDEX[object_id]

    assert object_id in GSO_OBJECT_NAME_TO_INDEX, f"Object {object_id} not found in GSO mapping data."
    return GSO_OBJECT_NAME_TO_INDEX[object_id]

def get_instance_name(object_id: int | str) -> str:
    class_id = get_instance_index(object_id)
    if class_id <= len(GSO_OBJECT_IDS):
        return MAPPING_GSO_DATA[GSO_OBJECT_IDS[class_id - 1]]["name"]
    return str(object_id)

def get_instance_color_map(object_id: int | str) -> tuple[int, int, int]:
    class_id = get_instance_index(object_id)
    hue = ((class_id - 1) * INSTANCE_COLOR_GOLDEN_ANGLE % 360.0) / 360.0
    rgb = hsv_to_rgb(np.array([hue, INSTANCE_COLOR_SATURATION, INSTANCE_COLOR_VALUE], dtype=np.float32))
    return tuple((rgb * 255).astype(np.uint8).tolist())


def get_semantic_color_lookup(simulation_data: dict) -> dict[tuple[int, int, int], tuple[int, int, int]]:
    classes = simulation_data["encoding"]["classes"]
    lookup = {}
    for object_id, object_data in simulation_data["objects"].items():
        lookup[tuple(classes[int(object_id) + 1])] = get_instance_color_map(object_data["model"])
    return lookup


def recolor_semantic_rgba(semantic_rgba: np.ndarray, color_lookup: dict[tuple[int, int, int], tuple[int, int, int]]) -> np.ndarray:
    for old_color, new_color in color_lookup.items():
        semantic_rgba[np.all(semantic_rgba[:, :, :3] == old_color, axis=-1), :3] = new_color
    return semantic_rgba


def get_object_material_colors(simulation_data: dict) -> dict[str, tuple[int, int, int]]:
    material_names = simulation_data["encoding"]["materials"]
    material_colors = {
        material_name: tuple(simulation_data["encoding"]["classes"][idx + 1])
        for idx, material_name in enumerate(material_names)
    }
    return {
        object_id: material_colors[object_data["description"]["material_group"]]
        for object_id, object_data in simulation_data["objects"].items()
    }


def get_mass_intensity(mass: float) -> float:
    scaled = np.clip((mass - GRAVITY_MASS_MIN_KG) / (GRAVITY_MASS_MAX_KG - GRAVITY_MASS_MIN_KG), 0.0, 1.0)
    boosted = scaled ** GRAVITY_INTENSITY_GAMMA
    return float(GRAVITY_MIN_VISIBLE_INTENSITY + (1.0 - GRAVITY_MIN_VISIBLE_INTENSITY) * boosted)


def add_gravity_legend(image: Image.Image) -> Image.Image:
    width, height = image.size
    bar_height = min(GRAVITY_LEGEND_HEIGHT, height - 2 * GRAVITY_LEGEND_MARGIN)
    x1 = width - GRAVITY_LEGEND_MARGIN
    x0 = max(4, x1 - GRAVITY_LEGEND_WIDTH)
    y0 = GRAVITY_LEGEND_MARGIN
    y1 = y0 + bar_height
    bar_x0 = x1 - GRAVITY_LEGEND_BAR_WIDTH

    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((x0, y0 - 10, x1, y1 + 10), fill=(0, 0, 0, 130))
    for row in range(bar_height):
        intensity = 1.0 - row / max(1, bar_height - 1)
        red = int(255 * (GRAVITY_MIN_VISIBLE_INTENSITY + (1.0 - GRAVITY_MIN_VISIBLE_INTENSITY) * intensity))
        draw.line((bar_x0, y0 + row, x1, y0 + row), fill=(red, 0, 0, 255))
    draw.rectangle((bar_x0, y0, x1, y1), outline=(255, 255, 255, 220))
    draw.text((x0 + 6, y0 - 2), f"{GRAVITY_MASS_MAX_KG:g} kg", fill=(255, 255, 255, 255))
    draw.text((x0 + 6, y1 - 10), f"{GRAVITY_MASS_MIN_KG:g} kg", fill=(255, 255, 255, 255))
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

   
def get_kinematics_data(simulations_json_file: str | Path) -> dict:
    return json.load(open(Path(simulations_json_file).with_name("simulation_kinematics.json")))

def get_object_velocity(frame_data: dict, next_frame_data: dict | None, frame_key: str, next_frame_key: str | None, object_id: str) -> np.ndarray:
    velocity = frame_data["objects"][object_id]["kinematics"]["v"]
    if velocity[0] is not None:
        return np.array(velocity, dtype=np.float32)
    if next_frame_data is None:
        return np.zeros(3, dtype=np.float32)
    center = np.array(frame_data["objects"][object_id]["obb"]["center"], dtype=np.float32)
    next_center = np.array(next_frame_data["objects"][object_id]["obb"]["center"], dtype=np.float32)
    dt = float(next_frame_key) - float(frame_key)
    return (next_center - center) / dt


def get_scene_flow_2d(scene_flow_rgba: np.ndarray, object_mask: np.ndarray, scene_flow_mask: np.ndarray, scene_flow_norm: float) -> np.ndarray:
    fg = (
        scene_flow_mask
        & ~np.all(object_mask[:, :, :3] == 0, axis=-1)
        & ~np.all(object_mask[:, :, :3] == 255, axis=-1)
    )
    if not np.any(fg):
        return np.zeros(2, dtype=np.float32)
    scene_flow = ((scene_flow_rgba[:, :, :3].astype(np.float32) / 255.0) * 2.0 - 1.0) * scene_flow_norm
    return scene_flow[fg, :2].mean(axis=0)


def get_optical_flow_color(flow_2d: np.ndarray, max_magnitude: float) -> np.ndarray:
    magnitude = np.linalg.norm(flow_2d)
    if magnitude == 0:
        return np.zeros(3, dtype=np.uint8)
    hue = (np.arctan2(flow_2d[1], flow_2d[0]) + np.pi) / (2 * np.pi)
    saturation = min(magnitude / max_magnitude, 1.0)
    rgb = hsv_to_rgb(np.array([hue, saturation, 1.0], dtype=np.float32))
    return (rgb * 255).astype(np.uint8)
   
def save_video(frames, new_path, video_name):
    os.makedirs(new_path, exist_ok=True)
    new_file_name = new_path / f"{video_name}.mp4"

    print(f"Saving video to {new_file_name} with {len(frames)} frames.")

    with imageio.get_writer(new_file_name, fps=30) as writer:
        for frame in frames:
            writer.append_data(np.array(frame.convert("RGB")))


##########################################
#               Segmentation             #
##########################################

def overlay_semantic_segmentation(simulations_json_file, rgb_frames, simulation_data=None):
    if simulation_data is None:
        simulation_data = json.load(open(simulations_json_file))
    semantic_frames = get_frames(simulations_json_file, "instances")
    color_lookup = get_semantic_color_lookup(simulation_data)
    assert len(rgb_frames) == len(semantic_frames), "Number of RGB frames and semantic frames do not match."

    blended_frames = []
    for rgb_frame, semantic_frame in tqdm(zip(rgb_frames, semantic_frames), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")
        semantic_rgba = np.array(Image.open(semantic_frame).convert("RGBA"))
        semantic_rgba = recolor_semantic_rgba(semantic_rgba, color_lookup)

        # Remove background pixels from the semantic image so only labeled regions overlay.
        background_mask = (
            np.all(semantic_rgba[:, :, :3] == 255, axis=-1)
            | np.all(semantic_rgba[:, :, :3] == 0, axis=-1)
        )
        semantic_rgba[background_mask, 3] = 0

        semantic_image = Image.fromarray(semantic_rgba, mode="RGBA")
        blended_image = Image.alpha_composite(rgb_image, semantic_image)

        blended_frames.append(blended_image)

    return blended_frames

def overlay_amodal_semantic_segmentation(simulations_json_file, rgb_frames, simulation_data, amodal_instances_frames, number_of_folders):
    semantic_frames = get_frames(simulations_json_file, "instances")
    assert len(rgb_frames) == len(semantic_frames), "Number of RGB frames and semantic frames do not match."

    blended_frames = []
    for i, rgb_frame in tqdm(enumerate(rgb_frames), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")

        composite_image = Image.new("RGBA", rgb_image.size, (0, 0, 0, 0))
        composite_np = np.array(composite_image).astype(np.uint16)

        for j in range(number_of_folders):
            amodal_frame = amodal_instances_frames[i + (j * len(rgb_frames))]
            object_id = amodal_frame.parent.name.split("_")[1]
            object_color = get_instance_color_map(simulation_data["objects"][object_id]["model"])
            amodal_rgba = Image.open(amodal_frame).convert("RGBA")
            amodal_np = np.array(amodal_rgba).astype(np.uint16)

            bg = (
                np.all(amodal_np[:, :, :3] == 255, axis=-1) |
                np.all(amodal_np[:, :, :3] == 0, axis=-1)
            )
            amodal_np[~bg, :3] = object_color
            amodal_np[bg, 3] = 0
            amodal_np[~bg, 3] = amodal_np[~bg, 3] // 2  # Reduce alpha for all object pixels

            fg = amodal_np[:, :, 3] > 0
            existing_fg = composite_np[:, :, 3] > 0

            new_only = fg & (~existing_fg)
            overlap = fg & existing_fg

            composite_np[new_only] = amodal_np[new_only]
            composite_np[overlap, :3] = np.clip(
                composite_np[overlap, :3] + amodal_np[overlap, :3] // 2,
                0, 255
            )
            composite_np[overlap, 3] = np.maximum(composite_np[overlap, 3], amodal_np[overlap, 3])

        composite_image = Image.fromarray(composite_np.astype(np.uint8), "RGBA")

        blended_image = Image.alpha_composite(rgb_image, composite_image)
        blended_frames.append(blended_image)

    return blended_frames
            

def overlay_material_segmentation(simulations_json_file, rgb_frames):
    material_frames = get_frames(simulations_json_file, "materials")
    assert len(rgb_frames) == len(material_frames), "Number of RGB frames and material frames do not match."

    blended_frames = []
    for rgb_frame, material_frame in tqdm(zip(rgb_frames, material_frames), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")
        material_rgba = np.array(Image.open(material_frame).convert("RGBA"))

        # Remove background pixels from the material image so only labeled regions overlay.
        background_mask = (
            np.all(material_rgba[:, :, :3] == 0, axis=-1)
        )
        material_rgba[background_mask, 3] = 0
        material_rgba[~background_mask, 3] = 128  # Set alpha to 128 for all material pixels for better visibility

        material_image = Image.fromarray(material_rgba, mode="RGBA")
        blended_image = Image.alpha_composite(rgb_image, material_image)

        blended_frames.append(blended_image)

    return blended_frames

def overlay_amodal_material_segmentation(simulations_json_file, rgb_frames, simulation_data, amodal_instances_frames, number_of_folders):
    material_frames = get_frames(simulations_json_file, "materials")
    semantic_frames = get_frames(simulations_json_file, "instances")
    object_material_colors = get_object_material_colors(simulation_data)
    object_instance_colors = {
        object_id: tuple(simulation_data["encoding"]["classes"][int(object_id) + 1])
        for object_id in simulation_data["objects"]
    }

    blended_frames = []
    n_frames = len(rgb_frames)

    for i, (rgb_frame, material_frame, semantic_frame) in tqdm(
        enumerate(zip(rgb_frames, material_frames, semantic_frames)),
        total=n_frames
    ):
        rgb_image = Image.open(rgb_frame).convert("RGBA")
        material_base = np.array(Image.open(material_frame).convert("RGBA"), dtype=np.uint16)
        semantic_base = np.array(Image.open(semantic_frame).convert("RGB"))

        composite_np = np.zeros((rgb_image.size[1], rgb_image.size[0], 4), dtype=np.uint16)

        for j in range(number_of_folders):
            object_id = amodal_instances_frames[i + j * n_frames].parent.name.split("_")[1]
            amodal_np = np.array(
                Image.open(amodal_instances_frames[i + j * n_frames]).convert("RGBA"),
                dtype=np.uint16
            )

            bg = (
                np.all(amodal_np[:, :, :3] == 255, axis=-1) |
                np.all(amodal_np[:, :, :3] == 0, axis=-1)
            )
            fg = ~bg
            visible_mask = fg & np.all(semantic_base == object_instance_colors[object_id], axis=-1)
            hidden_mask = fg & ~visible_mask

            obj_mat = np.zeros_like(material_base)
            obj_mat[visible_mask, :3] = material_base[visible_mask, :3]
            obj_mat[hidden_mask, :3] = object_material_colors[object_id]
            obj_mat[visible_mask, 3] = AMODAL_MATERIAL_VISIBLE_ALPHA
            obj_mat[hidden_mask, 3] = AMODAL_MATERIAL_HIDDEN_ALPHA

            existing_fg = composite_np[:, :, 3] > 0
            new_only = fg & (~existing_fg)
            overlap = fg & existing_fg

            composite_np[new_only] = obj_mat[new_only]

            composite_np[overlap, :3] = np.clip(
                composite_np[overlap, :3] + obj_mat[overlap, :3] // 2,
                0, 255
            )
            composite_np[overlap, 3] = np.maximum(
                composite_np[overlap, 3],
                obj_mat[overlap, 3]
            )

        composite_image = Image.fromarray(composite_np.astype(np.uint8), "RGBA")
        blended_image = Image.alpha_composite(rgb_image, composite_image)
        blended_frames.append(blended_image)

    return blended_frames


##########################################
#               Segmentation             #
##########################################

def overlay_instance_segmentation(simulations_json_file, rgb_frames, simulation_data=None):
    semantic_frames = get_frames(simulations_json_file, "instances")
    assert len(rgb_frames) == len(semantic_frames), "Number of RGB frames and semantic frames do not match."

    blended_frames = []
    for rgb_frame, semantic_frame in tqdm(zip(rgb_frames, semantic_frames), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")
        semantic_rgba = np.array(Image.open(semantic_frame).convert("RGBA"))

        # Remove background pixels from the semantic image so only labeled regions overlay.
        background_mask = (
            np.all(semantic_rgba[:, :, :3] == 255, axis=-1)
            | np.all(semantic_rgba[:, :, :3] == 0, axis=-1)
        )
        semantic_rgba[background_mask, 3] = 0

        semantic_image = Image.fromarray(semantic_rgba, mode="RGBA")
        blended_image = Image.alpha_composite(rgb_image, semantic_image)

        blended_frames.append(blended_image)

    return blended_frames

def overlay_amodal_instance_segmentation(simulations_json_file, rgb_frames, simulation_data, amodal_instances_frames, number_of_folders):
    semantic_frames = get_frames(simulations_json_file, "instances")

    blended_frames = []
    for i, (rgb_frame, semantic_frame) in tqdm(enumerate(zip(rgb_frames, semantic_frames)), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")

        composite_image = Image.new("RGBA", rgb_image.size, (0, 0, 0, 0))
        composite_np = np.array(composite_image).astype(np.uint16)

        for j in range(number_of_folders):
            amodal_rgba = Image.open(amodal_instances_frames[i + (j * len(rgb_frames))]).convert("RGBA")
            amodal_np = np.array(amodal_rgba).astype(np.uint16)

            bg = (
                np.all(amodal_np[:, :, :3] == 255, axis=-1) |
                np.all(amodal_np[:, :, :3] == 0, axis=-1)
            )
            amodal_np[bg, 3] = 0
            amodal_np[~bg, 3] = amodal_np[~bg, 3] // 2  # Reduce alpha for all object pixels

            fg = amodal_np[:, :, 3] > 0
            existing_fg = composite_np[:, :, 3] > 0

            new_only = fg & (~existing_fg)
            overlap = fg & existing_fg

            composite_np[new_only] = amodal_np[new_only]
            composite_np[overlap, :3] = np.clip(
                composite_np[overlap, :3] + amodal_np[overlap, :3] // 2,
                0, 255
            )
            composite_np[overlap, 3] = np.maximum(composite_np[overlap, 3], amodal_np[overlap, 3])

        composite_image = Image.fromarray(composite_np.astype(np.uint8), "RGBA")

        blended_image = Image.alpha_composite(rgb_image, composite_image)
        blended_frames.append(blended_image)

    return blended_frames

##########################################
#                Scene Flow              #
##########################################


def overlay_scene_flow(simulations_json_file, rgb_frames):
    scene_flow_frames = get_frames(simulations_json_file, "sceneflow")
    assert len(rgb_frames) == len(scene_flow_frames), "Number of RGB frames and scene flow frames do not match."

    blended_frames = []
    for rgb_frame, scene_flow_frame in tqdm(zip(rgb_frames, scene_flow_frames), total=len(rgb_frames)):
        rgb_image = Image.open(rgb_frame).convert("RGBA")
        scene_flow_rgba = np.array(Image.open(scene_flow_frame).convert("RGBA"))

        # --- COLOR MANIPULATION START ---
        # 1. Isolate RGB channels and normalize to 0.0 - 1.0 for matplotlib
        rgb_normalized = scene_flow_rgba[:, :, :3] / 255.0
        
        # 2. Convert to HSV
        hsv = rgb_to_hsv(rgb_normalized)

        # OPTION B: Multiply Saturation by 3 (Uncomment below to make colors 3x more vivid instead)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 3.0, 0.0, 1.0) 

        # 4. Convert back to RGB and map back to 0-255 uint8
        scene_flow_rgba[:, :, :3] = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
        # --- COLOR MANIPULATION END ---

        # Remove background pixels from the scene flow image so only labeled regions overlay.
        # (Note: Pure black and white have 0 saturation, so changing the hue won't alter them. 
        # This background mask will still work perfectly after the color manipulation.)
        background_mask = (
            np.all(scene_flow_rgba[:, :, :3] == 255, axis=-1)
            | np.all(scene_flow_rgba[:, :, :3] <= 20, axis=-1)
        )
        scene_flow_rgba[background_mask, 3] = 0

        scene_flow_image = Image.fromarray(scene_flow_rgba, mode="RGBA")
        blended_image = Image.alpha_composite(rgb_image, scene_flow_image)

        blended_frames.append(blended_image)

    return blended_frames


def overlay_optical_flow(simulations_json_file, rgb_frames, simulation_data=None):
    scene_flow_frames = get_frames(simulations_json_file, "sceneflow")
    assert len(rgb_frames) == len(scene_flow_frames), "Number of RGB frames and optical flow frames do not match."
    kinematics_data = get_kinematics_data(simulations_json_file)
    if simulation_data is None:
        simulation_data = json.load(open(simulations_json_file))
    frame_keys = list(kinematics_data["simulation"].keys())
    assert len(frame_keys) == len(rgb_frames), "Number of RGB frames and kinematics frames do not match."
    root_path = Path(simulations_json_file).parent
    scene_flow_norm = simulation_data["encoding"]["sceneflow_norm"]

    max_magnitude = 0.0
    for frame_idx, frame_key in enumerate(frame_keys):
        frame_data = kinematics_data["simulation"][frame_key]
        next_frame_key = frame_keys[frame_idx + 1] if frame_idx + 1 < len(frame_keys) else None
        next_frame_data = kinematics_data["simulation"][next_frame_key] if next_frame_key is not None else None
        rotation = compute_extrinsic_matrix(
            np.array(frame_data["camera"]["eye"], dtype=np.float32),
            np.array(frame_data["camera"]["at"], dtype=np.float32),
            np.array(frame_data["camera"]["up"], dtype=np.float32),
        )[:, :3]
        for object_id in frame_data["objects"]:
            velocity_world = get_object_velocity(frame_data, next_frame_data, frame_key, next_frame_key, object_id)
            flow_2d = (rotation @ velocity_world)[:2]
            max_magnitude = max(max_magnitude, float(np.linalg.norm(flow_2d)))
    max_magnitude = max(max_magnitude, 1e-6)

    blended_frames = []
    for frame_idx, (rgb_frame, scene_flow_frame, frame_key) in tqdm(
        enumerate(zip(rgb_frames, scene_flow_frames, frame_keys)),
        total=len(rgb_frames)
    ):
        frame_data = kinematics_data["simulation"][frame_key]
        next_frame_key = frame_keys[frame_idx + 1] if frame_idx + 1 < len(frame_keys) else None
        next_frame_data = kinematics_data["simulation"][next_frame_key] if next_frame_key is not None else None
        rotation = compute_extrinsic_matrix(
            np.array(frame_data["camera"]["eye"], dtype=np.float32),
            np.array(frame_data["camera"]["at"], dtype=np.float32),
            np.array(frame_data["camera"]["up"], dtype=np.float32),
        )[:, :3]
        rgb_np = np.array(Image.open(rgb_frame).convert("RGB"))
        scene_flow_rgba = np.array(Image.open(scene_flow_frame).convert("RGBA"))
        scene_flow_mask = np.max(scene_flow_rgba[:, :, :3], axis=-1) > OPTICAL_FLOW_BACKGROUND_THRESHOLD

        for object_id in frame_data["objects"]:
            velocity_world = get_object_velocity(frame_data, next_frame_data, frame_key, next_frame_key, object_id)
            flow_2d = (rotation @ velocity_world)[:2]
            object_mask = np.array(
                Image.open(root_path / "instances" / f"obj_{object_id}" / Path(rgb_frame).name).convert("RGBA")
            )
            fg = (
                scene_flow_mask
                & ~np.all(object_mask[:, :, :3] == 0, axis=-1)
                & ~np.all(object_mask[:, :, :3] == 255, axis=-1)
            )
            if np.linalg.norm(flow_2d) == 0:
                flow_2d = get_scene_flow_2d(scene_flow_rgba, object_mask, scene_flow_mask, scene_flow_norm)
            rgb_np[fg] = get_optical_flow_color(flow_2d, max_magnitude)

        blended_frames.append(Image.fromarray(rgb_np))

    return blended_frames


def overlay_collision(simulations_json_file, rgb_frames, alpha=2.9):
    collision_frames = get_frames(simulations_json_file, "force_collision")
    assert len(rgb_frames) == len(collision_frames), "Number of RGB frames and collision frames do not match."

    blended_frames = []

    for rgb_frame, collision_frame in tqdm(zip(rgb_frames, collision_frames), total=len(rgb_frames)):
        rgb = np.asarray(Image.open(rgb_frame).convert("RGB"), dtype=np.float32)
        collision = np.asarray(Image.open(collision_frame).convert("RGB"), dtype=np.float32)

        # use red channel as intensity mask
        mask = collision[..., 0:1] / 255.0

        # target color = strong red
        red = np.zeros_like(rgb)
        red[..., 0] = 255.0

        # blend original image toward red where mask is active
        out = rgb * (1 - alpha * mask) + red * (alpha * mask)
        out = np.clip(out, 0, 255).astype(np.uint8)

        blended_frames.append(add_gravity_legend(Image.fromarray(out)))

    return blended_frames


def overlay_gravity(simulations_json_file, rgb_frames, alpha=0.85):
    gravity_frames = get_frames(simulations_json_file, "force_pt_wise-gravity")
    instance_frames = get_frames(simulations_json_file, "instances")
    assert len(rgb_frames) == len(gravity_frames)
    assert len(rgb_frames) == len(instance_frames)
    simulation_data = json.load(open(simulations_json_file))
    object_colors = {
        object_id: tuple(simulation_data["encoding"]["classes"][int(object_id) + 1])
        for object_id in simulation_data["objects"]
    }

    blended_frames = []

    for rgb_frame, instance_frame in tqdm(zip(rgb_frames, instance_frames), total=len(rgb_frames)):
        rgb = np.asarray(Image.open(rgb_frame).convert("RGB"), dtype=np.float32)
        instances = np.asarray(Image.open(instance_frame).convert("RGB"))
        out = rgb.copy()

        for object_id, object_data in simulation_data["objects"].items():
            intensity = get_mass_intensity(float(object_data["mass"])) * alpha
            mask = np.all(instances == object_colors[object_id], axis=-1)
            out[mask] = out[mask] * (1.0 - intensity) + GRAVITY_RED * intensity

        out = np.clip(out, 0, 255).astype(np.uint8)

        blended_frames.append(add_gravity_legend(Image.fromarray(out)))

    return blended_frames


def overlay_stress(simulations_json_file, rgb_frames, alpha=10.0):

    # alpha = 10. (or whatever)
    # beta = norm(over - [0., 0., 0.]) / norm([1., 0., 0.])
    # color = color + alpha * beta * (over - color)

    collision_frames = get_frames(simulations_json_file, "force_defo_grad_wise-material")
    assert len(rgb_frames) == len(collision_frames)

    blended_frames = []

    for rgb_frame, collision_frame in tqdm(zip(rgb_frames, collision_frames), total=len(rgb_frames)):
        rgb = np.asarray(Image.open(rgb_frame).convert("RGB"), dtype=np.float32)
        overlay = np.asarray(Image.open(collision_frame).convert("RGB"), dtype=np.float32)

        # True Additive Blend: Just add the scaled overlay directly to the base image
        out = rgb + (overlay * alpha)
        
        # Clip to ensure values don't overflow past white (255)
        out = np.clip(out, 0, 255).astype(np.uint8)

        blended_frames.append(Image.fromarray(out))

    return blended_frames


def compute_video(simulations_json_file, map_name, output_path):
    rgb_frames = get_frames(simulations_json_file, "render")
    amodal_instances_frames, num_folders = get_all_instances_frames(simulations_json_file)
    simulation_data = json.load(open(simulations_json_file))

    ##########################################
    #                 RGB                    #
    ##########################################

    # print(f"Saved RGB video for {map_name}.")
    # frames_rgb_only = [Image.open(frame).convert("RGB") for frame in rgb_frames]
    # save_video(frames_rgb_only, output_path / map_name, "rgb_only")

    # ##########################################
    # #               Segmentation             #
    # ##########################################

    # print(f"Overlaying semantic segmentation on RGB frames for {map_name}...")
    # frames_semantic_segmentation = overlay_semantic_segmentation(simulations_json_file, rgb_frames)
    # save_video(frames_semantic_segmentation, output_path / map_name, "semantic_segmentation")

    # print(f"Overlaying AMODAL semantic segmentation on RGB frames for {map_name}...")
    # frames_amodal_semantic_segmentation = overlay_amodal_semantic_segmentation(simulations_json_file, rgb_frames, simulation_data, amodal_instances_frames, num_folders)
    # save_video(frames_amodal_semantic_segmentation, output_path / map_name, "amodal_semantic_segmentation")

    # print(f"Overlaying material segmentation on RGB frames for {map_name}...")
    # frames_materials = overlay_material_segmentation(simulations_json_file, rgb_frames)
    # save_video(frames_materials, output_path / map_name, "materials")

    # print(f"Overlaying AMODAL material segmentation on RGB frames for {map_name}...")
    # frames_materials = overlay_amodal_material_segmentation(simulations_json_file, rgb_frames, simulation_data, amodal_instances_frames, num_folders)
    # save_video(frames_materials, output_path / map_name, "amodal_materials")

    ##########################################
    #                Tracking                #
    ##########################################

    # print(f"Overlaying instance segmentation on RGB frames for {map_name}...")
    # frames_instances = overlay_instance_segmentation(simulations_json_file, rgb_frames)
    # save_video(frames_instances, output_path / map_name, "instances")

    # print(f"Overlaying AMODAL instance segmentation on RGB frames for {map_name}...")
    # frames_instances = overlay_amodal_instance_segmentation(simulations_json_file, rgb_frames, simulation_data, amodal_instances_frames, num_folders)
    # save_video(frames_instances, output_path / map_name, "amodal_instances")

    ##########################################
    #               Kinematics               #
    ##########################################

    # print("Copying depth video without modifications...")
    # shutil.copyfile(simulations_json_file.replace("simulation.json", "_fps-25_depth.mp4"),  Path("./out") / map_name / "depth.mp4")

    # print(f"Overlaying scene flow on RGB frames for {map_name}...")
    # frames_scene_flow = overlay_scene_flow(simulations_json_file, rgb_frames)
    # save_video(frames_scene_flow, output_path / map_name, "scene_flow")

    # print(f"Overlaying optical flow on RGB frames for {map_name}...")
    # frames_scene_flow = overlay_optical_flow(simulations_json_file, rgb_frames, simulation_data=simulation_data)
    # save_video(frames_scene_flow, output_path / map_name, "optical_flow")

    ##########################################
    #              Dynamics                  #
    ##########################################

    # print(f"Overlaying collision on RGB frames for {map_name}...")
    # frames_collision = overlay_collision(simulations_json_file, rgb_frames)
    # save_video(frames_collision, output_path / map_name, "collision")

    print(f"Overlaying gravity flow on RGB frames for {map_name}...")
    frames_gravity = overlay_gravity(simulations_json_file, rgb_frames)
    save_video(frames_gravity, output_path / map_name, "gravity")

    # print(f"Overlaying stress flow on RGB frames for {map_name}...")
    # frames_stress = overlay_stress(simulations_json_file, rgb_frames)
    # save_video(frames_stress, output_path / map_name, "stress")


def main():
    output_path = Path("/Users/sebastiancavada/Desktop/Compositional_physics_ECCV_26 (1)/newtphys.github.io/video_newtphys_maps_2/out")
    download_path = "/Users/sebastiancavada/Desktop/Compositional_physics_ECCV_26 (1)/newtphys.github.io/downloaded/dl3dv"
    simulations_json_files = glob.glob(os.path.join(download_path, "**", "simulation.json"), recursive=True)

    print(f"Found {len(simulations_json_files)} simulation.json files.")

    for simulations_json_file in simulations_json_files:
        path_name = simulations_json_file.split("/dl3dv/")[1].replace("/simulation.json", "")
        map_name = MAP_NAME.get(path_name, "unknown")
        if map_name == "unknown":
            print(f"Warning: Map name for path '{path_name}' not found in MAP_NAME dictionary. Using 'unknown' as map name.")

        print(f"Processing {simulations_json_file} - {map_name}...")
        compute_video(simulations_json_file, map_name, output_path)

if __name__ == "__main__":
    main()
