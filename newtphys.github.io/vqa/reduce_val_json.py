#!/usr/bin/env python3
"""Reduce the validation answers JSON by removing redundant fields."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_INPUT = Path(__file__).with_name("val_answer_run_28_general.json")
DEFAULT_OUTPUT = Path(__file__).with_name("val_answer_run_28_general.reduced.json")

OMITTED_CONSTANTS = {
    "task_type": "factual",
    "choice_type": "mcq",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shrink the validation answer JSON by removing constant fields and "
            "deriving mode from the idx suffix."
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


def mode_from_idx(idx: str) -> str:
    try:
        _, suffix = idx.rsplit("_", 1)
    except ValueError as exc:
        raise ValueError(f"Unexpected idx format: {idx!r}") from exc

    if suffix == "i":
        return "image-only"
    if suffix == "g":
        return "general"
    raise ValueError(f"Unexpected idx suffix in {idx!r}")


def build_reduced_payload(data: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[list[str]] = []

    for row_index, record in enumerate(data):
        if not isinstance(record, dict):
            raise ValueError(f"Record {row_index} is not a JSON object.")

        for field, expected in OMITTED_CONSTANTS.items():
            actual = record.get(field)
            if actual != expected:
                raise ValueError(
                    f"Cannot omit {field!r}: expected {expected!r}, found {actual!r}."
                )

        idx = require_string(record.get("idx"), field="idx")
        answer = require_string(record.get("answer"), field="answer")
        actual_mode = require_string(record.get("mode"), field="mode")
        expected_mode = mode_from_idx(idx)
        if actual_mode != expected_mode:
            raise ValueError(
                f"Cannot omit 'mode': expected {expected_mode!r}, found {actual_mode!r}."
            )

        records.append([idx, answer])

    return {
        "version": 1,
        "omitted_fields": {
            "task_type": "factual",
            "choice_type": "mcq",
            "mode": "derived from idx suffix: _i => image-only, _g => general",
        },
        "record_layout": ["idx", "answer"],
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
