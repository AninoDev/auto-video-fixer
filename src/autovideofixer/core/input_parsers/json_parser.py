"""JSON input-list parser (`--from-file-parser json`).

Accepts any of:
  - a JSON array of strings (paths): `["a.mp4", "b.mkv"]`
  - a JSON array of objects: `[{"input": "a.mp4", "output": "out/a.mp4",
    "recursive": true}, ...]` (only "input" is required)
  - an object wrapping either array form: `{"inputs": [...]}`

Unknown object keys are ignored. A missing "input" key on an object entry
raises ValueError.
"""

from __future__ import annotations

import json
from typing import Any

from autovideofixer.core.input_parsers.base import InputParser, InputSpec, register_parser


def _spec_from_entry(entry: Any) -> InputSpec:
    if isinstance(entry, str):
        return InputSpec(input_path=entry)
    if isinstance(entry, dict):
        if "input" not in entry or not entry["input"]:
            raise ValueError(f"json input-list entry missing required 'input' key: {entry!r}")
        return InputSpec(
            input_path=entry["input"],
            output_path=entry.get("output"),
            recursive=entry.get("recursive"),
        )
    raise ValueError(f"json input-list entry must be a string or object, got: {entry!r}")


@register_parser
class JsonInputParser(InputParser):
    name = "json"

    def parse(self, text: str, base_dir: str) -> list[InputSpec]:
        data = json.loads(text)
        if isinstance(data, dict):
            if "inputs" not in data:
                raise ValueError(
                    "json input-list object form must have an 'inputs' key, e.g. "
                    '{"inputs": [...]}'
                )
            entries = data["inputs"]
        elif isinstance(data, list):
            entries = data
        else:
            raise ValueError(
                "json input-list must be a JSON array or an object with an 'inputs' key"
            )
        if not isinstance(entries, list):
            raise ValueError("json input-list 'inputs' must be a JSON array")
        return [_spec_from_entry(entry) for entry in entries]
