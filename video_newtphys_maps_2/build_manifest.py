#!/usr/bin/env python3

import json
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "out"
MANIFEST_PATH = ROOT / "manifest.json"


def build_scene_record(scene_dir: Path) -> dict:
    available_files = sorted(
        file_path.name
        for file_path in scene_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".mp4"
    )

    return {
        "label": scene_dir.name,
        "path": f"./video_newtphys_maps_2/out/{quote(scene_dir.name)}",
        "available_files": available_files,
        "disabled_files": {},
    }


def main() -> None:
    scenes = [
        build_scene_record(scene_dir)
        for scene_dir in sorted(
            (path for path in OUT_DIR.iterdir() if path.is_dir()),
            key=lambda path: path.name.lower(),
        )
    ]

    payload = {
        "video_root": "./video_newtphys_maps_2/out",
        "scenes": scenes,
    }

    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"write {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
