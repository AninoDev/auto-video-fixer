"""Tests for the pluggable input-list parsers (core/input_parsers)."""

import pytest

from autovideofixer.core.input_parsers import InputSpec, get_parser, list_parsers
from autovideofixer.core.input_parsers.base import _PARSER_REGISTRY, register_parser


class TestRegistry:
    def test_list_parsers_has_all_builtins(self):
        names = set(list_parsers())
        assert names == {"shlex", "lines", "csv", "json"}

    def test_get_parser_returns_instance(self):
        p = get_parser("shlex")
        assert p.name == "shlex"

    def test_get_parser_unknown_name_raises_with_known_list(self):
        with pytest.raises(ValueError) as excinfo:
            get_parser("nope")
        msg = str(excinfo.value)
        assert "nope" in msg
        for name in ("shlex", "lines", "csv", "json"):
            assert name in msg

    def test_register_parser_is_reachable_via_get_parser(self):
        from autovideofixer.core.input_parsers.base import InputParser

        class _Dummy(InputParser):
            name = "dummy_test_parser"

            def parse(self, text, base_dir):
                return [InputSpec(input_path=text.strip())]

        register_parser(_Dummy)
        try:
            p = get_parser("dummy_test_parser")
            assert p.parse("x.mp4", ".") == [InputSpec(input_path="x.mp4")]
        finally:
            del _PARSER_REGISTRY["dummy_test_parser"]


class TestShlexParser:
    def parser(self):
        return get_parser("shlex")

    def test_quoted_paths_with_spaces(self):
        specs = self.parser().parse("a.mp4 \"b c.mp4\" 'd e.mp4'", ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b c.mp4", "d e.mp4"]

    def test_backslash_escaped_space(self):
        specs = self.parser().parse(r"a\ b.mp4 c.mp4", ".")
        assert [s.input_path for s in specs] == ["a b.mp4", "c.mp4"]

    def test_newlines_treated_as_whitespace(self):
        specs = self.parser().parse("a.mp4\nb.mp4\n\nc.mp4", ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b.mp4", "c.mp4"]

    def test_collapsed_whitespace_runs(self):
        specs = self.parser().parse("a.mp4    b.mp4\t\tc.mp4", ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b.mp4", "c.mp4"]

    def test_hash_not_treated_as_comment(self):
        specs = self.parser().parse("weird#file.mp4", ".")
        assert [s.input_path for s in specs] == ["weird#file.mp4"]

    def test_empty_text_returns_no_specs(self):
        assert self.parser().parse("", ".") == []


class TestLinesParser:
    def parser(self):
        return get_parser("lines")

    def test_basic_lines(self):
        specs = self.parser().parse("a.mp4\nb.mp4\n", ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b.mp4"]

    def test_skips_blank_lines(self):
        specs = self.parser().parse("a.mp4\n\n\nb.mp4\n", ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b.mp4"]

    def test_skips_comment_lines(self):
        specs = self.parser().parse("# a comment\na.mp4\n  # indented comment\nb.mp4\n", ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b.mp4"]

    def test_strips_one_matched_pair_of_quotes(self):
        specs = self.parser().parse("\"a b.mp4\"\n'c d.mp4'\nplain.mp4\n", ".")
        assert [s.input_path for s in specs] == ["a b.mp4", "c d.mp4", "plain.mp4"]

    def test_unmatched_quote_left_as_is(self):
        specs = self.parser().parse('"unmatched.mp4\n', ".")
        assert [s.input_path for s in specs] == ['"unmatched.mp4']

    def test_hash_inside_a_path_is_not_special_mid_line(self):
        # Only a LEADING '#' after stripping marks a comment line.
        specs = self.parser().parse("a#b.mp4\n", ".")
        assert [s.input_path for s in specs] == ["a#b.mp4"]


class TestCsvParser:
    def parser(self):
        return get_parser("csv")

    def test_positional_input_only(self):
        specs = self.parser().parse("a.mp4\nb.mp4\n", ".")
        assert specs == [InputSpec(input_path="a.mp4"), InputSpec(input_path="b.mp4")]

    def test_positional_with_output_and_recursive(self):
        specs = self.parser().parse("a.mp4,out/a.mp4,true\nb.mp4,,false\n", ".")
        assert specs == [
            InputSpec(input_path="a.mp4", output_path="out/a.mp4", recursive=True),
            InputSpec(input_path="b.mp4", output_path=None, recursive=False),
        ]

    def test_header_detected_case_insensitive(self):
        specs = self.parser().parse("Input,Output,Recursive\na.mp4,out.mp4,yes\n", ".")
        assert specs == [
            InputSpec(input_path="a.mp4", output_path="out.mp4", recursive=True),
        ]

    def test_header_with_reordered_optional_columns(self):
        # Header detection requires the first CELL to be "input"; the
        # optional output/recursive columns may still appear in either order.
        specs = self.parser().parse("input,recursive,output\na.mp4,true,out.mp4\n", ".")
        assert specs == [
            InputSpec(input_path="a.mp4", output_path="out.mp4", recursive=True),
        ]

    def test_blank_output_is_none(self):
        specs = self.parser().parse("input,output\na.mp4,\n", ".")
        assert specs[0].output_path is None

    def test_skips_fully_empty_rows(self):
        specs = self.parser().parse("a.mp4\n\n,,\nb.mp4\n", ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b.mp4"]

    def test_no_recursive_column_defaults_to_none(self):
        specs = self.parser().parse("a.mp4\n", ".")
        assert specs[0].recursive is None


class TestJsonParser:
    def parser(self):
        return get_parser("json")

    def test_array_of_strings(self):
        specs = self.parser().parse('["a.mp4", "b.mp4"]', ".")
        assert specs == [InputSpec(input_path="a.mp4"), InputSpec(input_path="b.mp4")]

    def test_array_of_objects(self):
        specs = self.parser().parse(
            '[{"input": "a.mp4", "output": "out/a.mp4", "recursive": true}]', "."
        )
        assert specs == [InputSpec(input_path="a.mp4", output_path="out/a.mp4", recursive=True)]

    def test_object_with_inputs_key_array_of_strings(self):
        specs = self.parser().parse('{"inputs": ["a.mp4", "b.mp4"]}', ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b.mp4"]

    def test_object_with_inputs_key_array_of_objects(self):
        specs = self.parser().parse('{"inputs": [{"input": "a.mp4"}]}', ".")
        assert specs == [InputSpec(input_path="a.mp4")]

    def test_mixed_array_of_strings_and_objects(self):
        specs = self.parser().parse('["a.mp4", {"input": "b.mp4"}]', ".")
        assert [s.input_path for s in specs] == ["a.mp4", "b.mp4"]

    def test_missing_input_key_raises(self):
        with pytest.raises(ValueError):
            self.parser().parse('[{"output": "a.mp4"}]', ".")

    def test_unknown_keys_ignored(self):
        specs = self.parser().parse('[{"input": "a.mp4", "extra": "ignored"}]', ".")
        assert specs == [InputSpec(input_path="a.mp4")]

    def test_object_without_inputs_key_raises(self):
        with pytest.raises(ValueError):
            self.parser().parse('{"foo": "bar"}', ".")
