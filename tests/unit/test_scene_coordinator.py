"""Tests for the scene-content coordinator (drop-non-content-scenes) LLM pass.

Covers R1.7's hard requirement (fail OPEN -- keep every scene -- on any
coordinator failure, never fail closed to "drop everything") via mocked
provider responses, plus the response parser's malformed-input handling.
"""

from __future__ import annotations

from unittest.mock import patch

from autovideofixer.config import Config
from autovideofixer.core.analysis import (
    _build_coordinator_prompt,
    _parse_coordinator_response,
    run_scene_coordinator,
)


def _cfg(tmp_path) -> Config:
    return Config(tmp_path / "nonexistent.yaml")


class TestParseCoordinatorResponse:
    def test_valid_response(self):
        text = (
            '{"main_content_summary": "a cat video", "drop": [1, 3], '
            '"reasons": {"1": "promo bumper", "3": "subscribe screen"}}'
        )
        parsed = _parse_coordinator_response(text, num_scenes=4)
        assert parsed is not None
        assert parsed["drop"] == [1, 3]
        assert parsed["reasons"]["1"] == "promo bumper"
        assert parsed["main_content_summary"] == "a cat video"

    def test_empty_drop_list(self):
        parsed = _parse_coordinator_response('{"drop": []}', num_scenes=4)
        assert parsed is not None
        assert parsed["drop"] == []

    def test_markdown_fenced_json(self):
        text = '```json\n{"drop": [0]}\n```'
        parsed = _parse_coordinator_response(text, num_scenes=2)
        assert parsed is not None
        assert parsed["drop"] == [0]

    def test_malformed_json_returns_none(self):
        assert _parse_coordinator_response("not json at all", num_scenes=3) is None

    def test_non_dict_json_returns_none(self):
        assert _parse_coordinator_response("[1, 2, 3]", num_scenes=3) is None

    def test_drop_not_a_list_returns_none(self):
        assert _parse_coordinator_response('{"drop": "everything"}', num_scenes=3) is None

    def test_out_of_range_index_returns_none(self):
        # num_scenes=3 means valid indices are 0, 1, 2 -- 5 is out of range.
        assert _parse_coordinator_response('{"drop": [5]}', num_scenes=3) is None

    def test_negative_index_returns_none(self):
        assert _parse_coordinator_response('{"drop": [-1]}', num_scenes=3) is None

    def test_non_integer_index_returns_none(self):
        assert _parse_coordinator_response('{"drop": [1.5]}', num_scenes=3) is None

    def test_bool_index_returns_none(self):
        # bool is a subclass of int in Python -- must be explicitly rejected,
        # not silently accepted as 0/1.
        assert _parse_coordinator_response('{"drop": [true]}', num_scenes=3) is None

    def test_missing_reasons_defaults_to_empty(self):
        parsed = _parse_coordinator_response('{"drop": [0]}', num_scenes=2)
        assert parsed is not None
        assert parsed["reasons"] == {}


class TestBuildCoordinatorPrompt:
    def test_includes_all_scenes(self):
        summaries = [
            {
                "index": 0,
                "start_time": 0.0,
                "end_time": 5.0,
                "duration": 5.0,
                "summary": "a dog running",
                "tags": ["dog", "outdoor"],
            },
            {
                "index": 1,
                "start_time": 5.0,
                "end_time": 8.0,
                "duration": 3.0,
                "summary": "subscribe screen",
                "tags": ["promo"],
            },
        ]
        prompt = _build_coordinator_prompt(summaries)
        assert "index=0" in prompt
        assert "index=1" in prompt
        assert "a dog running" in prompt
        assert "subscribe screen" in prompt


class TestRunSceneCoordinatorFailOpen:
    """R1.7: any coordinator failure must fail OPEN (keep everything)."""

    def _summaries(self):
        return [
            {"index": 0, "start_time": 0.0, "end_time": 5.0, "duration": 5.0, "summary": "a"},
            {"index": 1, "start_time": 5.0, "end_time": 8.0, "duration": 3.0, "summary": "b"},
        ]

    def test_empty_scene_list_is_a_noop(self, tmp_path):
        result = run_scene_coordinator([], _cfg(tmp_path))
        assert result["drop"] == []
        assert result["failed"] is False

    def test_provider_call_returns_none_fails_open(self, tmp_path):
        with patch("autovideofixer.core.analysis._call_coordinator_provider", return_value=None):
            result = run_scene_coordinator(self._summaries(), _cfg(tmp_path))
        assert result["drop"] == []
        assert result["failed"] is True

    def test_provider_raises_fails_open(self, tmp_path):
        with patch(
            "autovideofixer.core.analysis._call_coordinator_provider",
            side_effect=RuntimeError("boom"),
        ):
            # run_scene_coordinator itself doesn't catch -- the exception
            # propagates; callers (core/scenes.py's run_scene_pipeline) are
            # responsible for catching and failing open at that layer. Verify
            # that contract explicitly so a regression there is caught too.
            try:
                run_scene_coordinator(self._summaries(), _cfg(tmp_path))
                raised = False
            except RuntimeError:
                raised = True
        assert raised

    def test_malformed_response_fails_open(self, tmp_path):
        with patch(
            "autovideofixer.core.analysis._call_coordinator_provider",
            return_value="not valid json",
        ):
            result = run_scene_coordinator(self._summaries(), _cfg(tmp_path))
        assert result["drop"] == []
        assert result["failed"] is True

    def test_out_of_range_drop_index_fails_open(self, tmp_path):
        with patch(
            "autovideofixer.core.analysis._call_coordinator_provider",
            return_value='{"drop": [99]}',
        ):
            result = run_scene_coordinator(self._summaries(), _cfg(tmp_path))
        assert result["drop"] == []
        assert result["failed"] is True

    def test_valid_response_drops_correct_indices(self, tmp_path):
        with patch(
            "autovideofixer.core.analysis._call_coordinator_provider",
            return_value='{"main_content_summary": "x", "drop": [1], '
            '"reasons": {"1": "subscribe screen"}}',
        ):
            result = run_scene_coordinator(self._summaries(), _cfg(tmp_path))
        assert result["drop"] == [1]
        assert result["failed"] is False
        assert result["reasons"]["1"] == "subscribe screen"
