"""CSV input-list parser (`--from-file-parser csv`).

Columns are positional by default: `input[,output][,recursive]`. If the
first row's first cell is exactly "input" (case-insensitive), it's treated
as a header row and columns are looked up by name (`input`/`output`/
`recursive`) in whatever order they appear, so a manifest can omit or
reorder the optional columns.
"""

from __future__ import annotations

import csv
import io

from autovideofixer.core.input_parsers.base import InputParser, InputSpec, register_parser

_TRUE_VALUES = {"true", "1", "yes", "y"}
_FALSE_VALUES = {"false", "0", "no", "n", ""}


def _parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in _TRUE_VALUES:
        return True
    if v in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value in csv 'recursive' column: {value!r}")


@register_parser
class CsvInputParser(InputParser):
    name = "csv"

    def parse(self, text: str, base_dir: str) -> list[InputSpec]:
        reader = csv.reader(io.StringIO(text))
        rows = [row for row in reader if any(cell.strip() for cell in row)]
        if not rows:
            return []

        header: list[str] | None = None
        if rows[0] and rows[0][0].strip().lower() == "input":
            header = [c.strip().lower() for c in rows[0]]
            rows = rows[1:]

        specs: list[InputSpec] = []
        for row in rows:
            if header is not None:
                cells = dict(zip(header, row, strict=False))
                input_path = cells.get("input", "").strip()
                output = cells.get("output", "").strip() or None
                recursive_cell = cells.get("recursive", "").strip()
            else:
                input_path = row[0].strip() if len(row) > 0 else ""
                output = (row[1].strip() or None) if len(row) > 1 else None
                recursive_cell = row[2].strip() if len(row) > 2 else ""

            if not input_path:
                continue
            recursive = _parse_bool(recursive_cell) if recursive_cell else None
            specs.append(InputSpec(input_path=input_path, output_path=output, recursive=recursive))
        return specs
