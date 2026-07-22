"""Auto Video Fixer - Pluggable input-list parsers.

Each parser turns the text of a `--from-file` list file into a list of
`InputSpec`s (input path, optional output path, optional recursive
override). Selected by name via `--from-file-parser` or an inline
`name:PATH` override on `--from-file` -- see AGENTS.md's "Input file lists"
section for the full CLI-side contract.

To add a new parser: subclass `InputParser` (see `base.py`), set a unique
class-level `name`, decorate the class with `@register_parser` (or call
`register_parser(cls)` after defining it), and import the module here so
registration runs at package-import time -- same pattern as
`core/stages/__init__.py`.
"""

from autovideofixer.core.input_parsers.base import (
    InputParser,
    InputSpec,
    get_parser,
    list_parsers,
    register_parser,
)
from autovideofixer.core.input_parsers.csv_parser import CsvInputParser
from autovideofixer.core.input_parsers.json_parser import JsonInputParser
from autovideofixer.core.input_parsers.lines_parser import LinesInputParser
from autovideofixer.core.input_parsers.shlex_parser import ShlexInputParser

__all__ = [
    "InputParser",
    "InputSpec",
    "get_parser",
    "list_parsers",
    "register_parser",
    "ShlexInputParser",
    "LinesInputParser",
    "CsvInputParser",
    "JsonInputParser",
]
