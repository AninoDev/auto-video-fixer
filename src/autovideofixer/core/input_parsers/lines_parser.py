"""Line-oriented input-list parser (`--from-file-parser lines`).

One path per line -- the classic manifest style: blank lines and `#`-comment
lines are skipped, and one matched pair of surrounding quotes is stripped if
present (so a path containing leading/trailing whitespace can still be
expressed by quoting it).
"""

from __future__ import annotations

from autovideofixer.core.input_parsers.base import InputParser, InputSpec, register_parser


def _strip_matched_quotes(s: str) -> str:
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


@register_parser
class LinesInputParser(InputParser):
    name = "lines"

    def parse(self, text: str, base_dir: str) -> list[InputSpec]:
        specs: list[InputSpec] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            line = _strip_matched_quotes(line)
            specs.append(InputSpec(input_path=line))
        return specs
