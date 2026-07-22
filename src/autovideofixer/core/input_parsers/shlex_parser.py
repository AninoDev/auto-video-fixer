"""shlex-based input-list parser -- the default (`--from-file-parser shlex`).

Splits on whitespace (including newlines) while honoring quotes and
backslash escapes, exactly like a POSIX shell command line would. This is
deliberately the default because it's what a Dolphin (or most file
managers') drag-and-drop-onto-a-terminal paste produces: a run of
whitespace-or-newline-separated, individually-quoted paths.
"""

from __future__ import annotations

import shlex

from autovideofixer.core.input_parsers.base import InputParser, InputSpec, register_parser


@register_parser
class ShlexInputParser(InputParser):
    name = "shlex"

    def parse(self, text: str, base_dir: str) -> list[InputSpec]:
        # posix=True: quotes are consumed (not kept literal), backslash
        # escapes work, and any run of whitespace (including newlines) is a
        # separator. comments=False (shlex.split's default): '#' is a valid
        # path character and must not truncate a token.
        tokens = shlex.split(text, comments=False, posix=True)
        return [InputSpec(input_path=tok) for tok in tokens]
