#!/usr/bin/env python3
"""Reduce the large evaluation JSON into a compact, still-readable format.

The reduced format removes redundant constants, stores each simulation path once,
stores only the path segment between ``dl3dv/`` and ``/render/``, and keeps only
PNG basenames in each record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path(__file__).with_name("test_run_28_general-all_karo_10K.json")
DEFAULT_OUTPUT = DEFAULT_INPUT.with_name(f"{DEFAULT_INPUT.stem}.reduced.json")

DL3DV_MARKER = "/dl3dv/"
SIMULATION_SUFFIX = "/simulation.json"
RENDER_MARKER = "/render/"

OMITTED_CONSTANTS = {
    "source": "simulation",
    "description": None,
    "choice_type": "mcq",
    "split": "val",
}

# Hardcoded exclusions for now. The row counts were measured on the original file.
EXCLUDED_QUESTION_IDS = {
    "F_OCCLUSION_PERCENTAGE_OBJECT",
    "F_MATERIAL_IDENTIFICATION_OBJECT_LEVEL_3",
    "F_CAMERA_ZOOM_BEHAVIOR",
    "F_FOCAL_LENGTH_CLASS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shrink the dataset JSON by removing redundant fields and factoring "
            "shared simulation metadata into lookup tables."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input JSON file. Defaults to {DEFAULT_INPUT.name}.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file. Defaults to {DEFAULT_OUTPUT.name}.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        help="Pretty-print the output JSON instead of writing it minified.",
    )
    return parser.parse_args()


def require_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Expected {field!r} to be a string, got {type(value).__name__}.")
    return value


def require_string_list(value: Any, *, field: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Expected {field!r} to be a list of strings.")
    return value


def extract_relative_simulation_path(simulation_id: str) -> str:
    if DL3DV_MARKER not in simulation_id or not simulation_id.endswith(SIMULATION_SUFFIX):
        raise ValueError(f"Unexpected simulation_id format: {simulation_id!r}")
    return simulation_id.split(DL3DV_MARKER, 1)[1][: -len(SIMULATION_SUFFIX)]


def extract_relative_render_path_and_frame_id(file_name: str) -> tuple[str, int]:
    if DL3DV_MARKER not in file_name or RENDER_MARKER not in file_name:
        raise ValueError(f"Unexpected file_name format: {file_name!r}")

    relative = file_name.split(DL3DV_MARKER, 1)[1]
    relative_path, png_name = relative.rsplit(RENDER_MARKER, 1)
    relative_path = relative_path.rstrip("/")

    if not png_name:
        raise ValueError(f"Missing PNG filename in {file_name!r}")

    frame_stem, extension = png_name.rsplit(".", 1)
    if extension.lower() != "png":
        raise ValueError(f"Expected a PNG filename, got {file_name!r}")

    return relative_path, int(frame_stem)

def expected_mode_from_png_count(png_count: int) -> str:
    return "image-only" if png_count == 1 else "general"


def validate_omitted_fields(record: dict[str, Any], png_count: int) -> None:
    for field, expected in OMITTED_CONSTANTS.items():
        actual = record.get(field)
        if actual != expected:
            raise ValueError(
                f"Cannot omit {field!r}: expected {expected!r}, found {actual!r}."
            )

    actual_mode = record.get("mode")
    expected_mode = expected_mode_from_png_count(png_count)
    if actual_mode != expected_mode:
        raise ValueError(
            f"Cannot omit 'mode': expected {expected_mode!r}, found {actual_mode!r}."
        )


def build_reduced_payload(data: list[dict[str, Any]]) -> dict[str, Any]:
    simulations: list[list[str]] = []
    simulation_index_by_path: dict[str, int] = {}

    question_meta: list[list[str]] = []
    question_index_by_id: dict[str, int] = {}

    records: list[list[Any]] = []

    for row_index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {row_index} is not a JSON object.")

        question_id = require_string(record.get("question_id"), field="question_id")
        if question_id in EXCLUDED_QUESTION_IDS:
            continue

        simulation_id = require_string(record.get("simulation_id"), field="simulation_id")
        relative_path = extract_relative_simulation_path(simulation_id)

        scene = require_string(record.get("scene"), field="scene")
        simulation_index = simulation_index_by_path.get(relative_path)
        if simulation_index is None:
            simulation_index = len(simulations)
            simulation_index_by_path[relative_path] = simulation_index
            simulations.append([scene, relative_path])
        elif simulations[simulation_index][0] != scene:
            raise ValueError(
                f"Simulation path {relative_path!r} maps to multiple scenes."
            )

        file_names = require_string_list(record.get("file_name"), field="file_name")
        frame_ids: list[int] = []
        for file_name in file_names:
            frame_relative_path, frame_id = extract_relative_render_path_and_frame_id(file_name)
            if frame_relative_path != relative_path:
                raise ValueError(
                    "file_name path does not match simulation_id path "
                    f"for record {row_index}."
                )
            frame_ids.append(frame_id)

        validate_omitted_fields(record, len(frame_ids))

        category = require_string(record.get("category"), field="category")
        sub_category = require_string(record.get("sub_category"), field="sub_category")

        question_index = question_index_by_id.get(question_id)
        if question_index is None:
            question_index = len(question_meta)
            question_index_by_id[question_id] = question_index
            question_meta.append([question_id, category, sub_category])
        else:
            existing_question_id, existing_category, existing_sub_category = question_meta[
                question_index
            ]
            if (
                existing_question_id != question_id
                or existing_category != category
                or existing_sub_category != sub_category
            ):
                raise ValueError(f"question_id {question_id!r} maps inconsistently.")

        question = require_string(record.get("question"), field="question")
        idx = record.get("idx")
        if not isinstance(idx, (str, int)):
            raise ValueError(f"Expected 'idx' to be a string or int in record {row_index}.")

        records.append([simulation_index, frame_ids, question, idx, question_index])

    return {
        "version": 1,
        "path_layout": {
            "simulation_relative_root": "dl3dv/",
            "simulation_suffix": "simulation.json",
            "render_dir": "render",
            "frame_name_format": "zero-pad frame_id to 6 digits and append .png",
        },
        "omitted_fields": {
            "source": "simulation",
            "description": None,
            "choice_type": "mcq",
            "split": "val",
            "mode": "derived from frame_ids count: 1 => image-only, >1 => general",
        },
        "excluded_question_ids": sorted(EXCLUDED_QUESTION_IDS),
        "simulation_layout": ["scene", "relative_path"],
        "question_layout": ["question_id", "category", "sub_category"],
        "record_layout": [
            "simulation_index",
            "frame_ids",
            "question",
            "idx",
            "question_meta_index",
        ],
        "simulations": simulations,
        "question_meta": question_meta,
        "records": records,
    }


def dump_json(payload: dict[str, Any], output_path: Path, *, indent: int | None) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        if indent is None:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()

    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected the input JSON to be a list, got {type(data).__name__}.")

    payload = build_reduced_payload(data)
    dump_json(payload, output_path, indent=args.indent)

    original_size = input_path.stat().st_size
    reduced_size = output_path.stat().st_size
    reduction = (1.0 - (reduced_size / original_size)) * 100 if original_size else 0.0

    print(f"Input:   {input_path}")
    print(f"Output:  {output_path}")
    print(f"Before:  {human_size(original_size)}")
    print(f"After:   {human_size(reduced_size)}")
    print(f"Saved:   {reduction:.2f}%")


if __name__ == "__main__":
    main()
