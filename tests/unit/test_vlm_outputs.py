"""§ 26.1: generated prompts / json_schema from the enabled outputs, parsing, validation."""

from __future__ import annotations

import itertools

import pytest

from autovideofixer.config import Config
from autovideofixer.core import analysis as an
from autovideofixer.core.vlm_outputs import (
    DEFAULT_MEDIA_TYPE_CHOICES,
    DETAIL_LEVELS,
    OutputsConfig,
    build_response_schema,
    build_system_prompt,
    build_user_prompt,
    normalize_media_type,
    parse_fields,
)

LEGACY_SYSTEM = (
    "You are a video analysis assistant. Analyze the provided frames from a video "
    "and provide: a brief summary of the content (1-2 sentences), relevant tags "
    "(comma-separated keywords), objects detected (comma-separated), and an "
    "appropriate content rating (G, PG, PG-13, R). Respond in valid JSON format "
    'with keys: "summary", "tags", "objects", "rating".'
)
LEGACY_USER = (
    "Analyze these video frames and describe the content. "
    "Provide a summary, tags, detected objects, and content rating."
)


class TestPromptGeneration:
    def test_brief_without_optional_outputs_is_byte_identical_to_legacy(self):
        o = OutputsConfig(media_type=False)
        assert build_system_prompt(o) == LEGACY_SYSTEM
        assert build_user_prompt(o) == LEGACY_USER
        assert an._VLM_SYSTEM_PROMPT == LEGACY_SYSTEM and an._VLM_USER_PROMPT == LEGACY_USER

    @pytest.mark.parametrize(
        "media,reasoning,detail",
        list(itertools.product([False, True], [False, True], DETAIL_LEVELS)),
    )
    def test_every_combination(self, media, reasoning, detail):
        o = OutputsConfig(media_type=media, rating_reasoning=reasoning, detail=detail)
        system, user = build_system_prompt(o), build_user_prompt(o)
        keys = ["summary", "tags", "objects"] + (["media_type"] if media else []) + ["rating"]
        keys += ["rating_reasoning"] if reasoning else []
        assert o.keys == tuple(keys)
        # The prompt lists exactly the enabled keys, in order, as its JSON key list.
        listed = ", ".join(f'"{k}"' for k in keys)
        assert f"with keys: {listed}." in system or f"keys: {listed}." in system
        assert ("media_type" in system) == media
        assert ("rating_reasoning" in system) == reasoning
        assert ("media type" in user) == media
        assert ("reasoning for the rating" in user) == reasoning
        if reasoning:  # asks for the justification after the content, as the last key
            assert system.index('"rating"') < system.index('"rating_reasoning"')
            assert "last two keys" in system
        # Longer detail = a longer instruction, never a shorter one.
        if detail != "brief":
            assert len(system) > len(build_system_prompt(OutputsConfig(media_type=media)))

    def test_detail_levels_grow(self):
        lengths = [len(build_system_prompt(OutputsConfig(detail=d))) for d in DETAIL_LEVELS]
        assert lengths == sorted(lengths) and len(set(lengths)) == len(lengths)

    def test_media_choices_listed_in_prompt(self):
        o = OutputsConfig(media_type_choices=("trail_cam", "other"))
        s = build_system_prompt(o)
        assert "exactly one of: trail_cam, other" in s and "real_footage" not in s

    def test_other_added_when_missing(self):
        o = OutputsConfig(media_type_choices=("a", "b"))
        assert o.schema_media_types == ["a", "b", "other"]


class TestSchema:
    @pytest.mark.parametrize(
        "media,reasoning", list(itertools.product([False, True], [False, True]))
    )
    def test_keys_and_order_follow_outputs(self, media, reasoning):
        o = OutputsConfig(media_type=media, rating_reasoning=reasoning)
        schema = build_response_schema(o)
        assert list(schema["properties"]) == list(o.keys)
        assert schema["required"] == list(o.keys)
        if reasoning:
            assert list(schema["properties"])[-1] == "rating_reasoning"
        if media:
            assert schema["properties"]["media_type"]["enum"] == list(DEFAULT_MEDIA_TYPE_CHOICES)
        assert schema["properties"]["rating"]["enum"] == ["G", "PG", "PG-13", "R"]

    def test_schema_reaches_payload_in_order(self):
        o = OutputsConfig(rating_reasoning=True)
        compat = {"system_content": "string", "response_format": "json_schema", "extra_body": {}}
        p = an._build_openai_payload("m", "S", "U", [], 10, compat, build_response_schema(o))
        schema = p["response_format"]["json_schema"]["schema"]
        assert list(schema["properties"]) == [
            "summary", "tags", "objects", "media_type", "rating", "rating_reasoning",
        ]  # fmt: skip


class TestConfigValidation:
    def test_defaults(self):
        o = OutputsConfig.from_config({})
        assert o.media_type and not o.rating_reasoning and o.detail == "brief"
        assert o.media_type_choices == DEFAULT_MEDIA_TYPE_CHOICES

    def test_real_default_config_validates(self, tmp_path):
        cfg = Config(tmp_path / "none.yaml")
        o = OutputsConfig.from_config(cfg.get("analysis", "vlm", default={}))
        assert o == OutputsConfig()

    @pytest.mark.parametrize(
        "vlm,match",
        [
            ({"detail": "verbose"}, "analysis.vlm.detail 'verbose'.*brief"),
            ({"detail": 3}, "analysis.vlm.detail"),
            ({"outputs": {"media_type": "yes"}}, "outputs.media_type.*true or false"),
            ({"outputs": {"nope": True}}, "Unknown key.*nope.*media_type"),
            ({"outputs": ["media_type"]}, "outputs.*mapping"),
            ({"media_type_choices": []}, "media_type_choices.*non-empty"),
            ({"media_type_choices": "real_footage"}, "media_type_choices"),
            ({"media_type_choices": ["a", 3]}, "media_type_choices"),
            ({"media_type_choices": ["a", "!!!"]}, "entry '!!!'"),
        ],
    )
    def test_errors_name_the_key(self, vlm, match):
        with pytest.raises(ValueError, match=match):
            OutputsConfig.from_config(vlm)

    def test_choices_dedupe_by_normalised_form(self):
        o = OutputsConfig.from_config({"media_type_choices": ["Cartoon", "cartoon", "Real Life"]})
        assert o.media_type_choices == ("Cartoon", "Real Life")

    def test_detail_is_case_insensitive(self):
        assert OutputsConfig.from_config({"detail": " Detailed "}).detail == "detailed"

    def test_run_vlm_analysis_rejects_bad_outputs_before_sampling(self, tmp_path):
        cfg = Config(tmp_path / "none.yaml")
        cfg.set("bogus", "analysis", "vlm", "detail")
        with pytest.raises(ValueError, match="analysis.vlm.detail"):
            an.VideoAnalyzer(cfg).run_vlm_analysis("does-not-exist.mp4")


class TestParsing:
    @pytest.mark.parametrize(
        "raw,expected,kept_raw",
        [
            ("real_footage", "real_footage", None),
            ("Real Footage", "real_footage", "Real Footage"),
            ("3D Rendered", "3d_rendered", "3D Rendered"),
            ("2d-digital", "2d_digital", "2d-digital"),
            ("  MIXED ", "mixed", "MIXED"),
            ("cartoon", "other", "cartoon"),
            ("", None, None),
            (None, None, None),
        ],
    )
    def test_normalize_media_type(self, raw, expected, kept_raw):
        got, original = normalize_media_type(raw)
        assert got == expected and original == kept_raw

    def test_full_reply(self):
        text = (
            '{"summary": "s", "tags": "a, b", "objects": ["x"], "media_type": "3D Rendered", '
            '"rating": "PG", "rating_reasoning": "mild peril", "style_notes": "cel shaded", '
            '"counts": {"people": 2}}'
        )
        r = an._parse_vlm_response(text)
        assert r["tags"] == ["a", "b"] and r["objects"] == ["x"]
        assert r["media_type"] == "3d_rendered" and r["rating"] == "PG"
        assert r["rating_reasoning"] == "mild peril"
        assert r["extra"] == {
            "style_notes": "cel shaded",
            "counts": {"people": 2},
            "media_type_raw": "3D Rendered",
        }

    def test_clean_reply_has_empty_extra(self):
        r = an._parse_vlm_response('{"summary": "s", "media_type": "real_footage"}')
        assert r["media_type"] == "real_footage" and r["extra"] == {}
        assert r["rating_reasoning"] is None

    def test_unlisted_type_is_other_with_raw_kept(self):
        r = an._parse_vlm_response('{"summary": "s", "media_type": "claymation"}')
        assert r["media_type"] == "other" and r["extra"] == {"media_type_raw": "claymation"}

    def test_custom_choices(self):
        r = an._parse_vlm_response('{"media_type": "trail cam"}', ("trail_cam", "other"))
        assert r["media_type"] == "trail_cam"

    def test_old_style_reply_still_parses(self):
        r = an._parse_vlm_response('{"summary": "s", "tags": "a", "objects": "b", "rating": "G"}')
        assert r["media_type"] is None and r["rating_reasoning"] is None and r["extra"] == {}

    def test_non_json_fallback_has_all_keys(self):
        r = an._parse_vlm_response("just words")
        assert r["summary"] == "just words" and r["extra"] == {} and r["media_type"] is None

    def test_parse_fields_non_string_reasoning(self):
        assert parse_fields({"rating_reasoning": 5})["rating_reasoning"] == "5"
