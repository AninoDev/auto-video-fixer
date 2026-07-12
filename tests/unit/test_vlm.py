"""Tests for VLM (Vision Language Model) integration."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.core.analysis import (
    _call_custom_api,
    _call_ollama,
    _call_openai,
    _extract_sample_frames,
    _frames_to_base64,
    _parse_vlm_response,
)


class TestParseVlmResponse:
    """Test VLM response parsing."""

    def test_parse_valid_json(self):
        """Test parsing a valid JSON response."""
        response = json.dumps(
            {
                "summary": "A cat sleeping on a couch",
                "tags": "cat, indoor, relaxing",
                "objects": "cat, couch, pillow",
                "rating": "G",
            }
        )
        result = _parse_vlm_response(response)
        assert result["summary"] == "A cat sleeping on a couch"
        assert result["tags"] == ["cat", "indoor", "relaxing"]
        assert result["objects"] == ["cat", "couch", "pillow"]
        assert result["rating"] == "G"

    def test_parse_json_with_code_fences(self):
        """Test parsing JSON wrapped in markdown code fences."""
        response = (
            '```json\n{"summary": "test", "tags": "a,b", "objects": "c", "rating": "PG"}\n```'
        )
        result = _parse_vlm_response(response)
        assert result["summary"] == "test"
        assert result["rating"] == "PG"

    def test_parse_tags_as_list(self):
        """Test parsing tags when they are a JSON array."""
        response = json.dumps(
            {
                "summary": "test",
                "tags": ["action", "comedy"],
                "objects": [],
                "rating": "PG-13",
            }
        )
        result = _parse_vlm_response(response)
        assert result["tags"] == ["action", "comedy"]

    def test_parse_invalid_json_fallback(self):
        """Test that invalid JSON falls back to treating text as summary."""
        response = "This is just plain text, not JSON."
        result = _parse_vlm_response(response)
        assert result["summary"] == "This is just plain text, not JSON."
        assert result["tags"] == []
        assert result["objects"] == []

    def test_parse_empty_response(self):
        """Test parsing an empty response."""
        result = _parse_vlm_response("")
        assert result["summary"] == ""
        assert result["tags"] == []
        assert result["objects"] == []

    def test_parse_missing_keys(self):
        """Test parsing a response missing some expected keys."""
        response = json.dumps({"summary": "only summary"})
        result = _parse_vlm_response(response)
        assert result["summary"] == "only summary"
        assert result["tags"] == []
        assert result["objects"] == []
        assert result["rating"] is None


class TestFramesToBase64:
    """Test frame-to-base64 conversion."""

    def test_empty_list(self, tmp_path):
        """Test with no frames."""
        assert _frames_to_base64([]) == []

    def test_single_frame(self, tmp_path):
        """Test with a single frame file."""
        frame = tmp_path / "test.jpg"
        frame.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Fake JPEG
        result = _frames_to_base64([str(frame)])
        assert len(result) == 1
        assert len(result[0]) > 0

    def test_multiple_frames(self, tmp_path):
        """Test with multiple frame files."""
        frames = []
        for i in range(3):
            f = tmp_path / f"frame_{i}.jpg"
            f.write_bytes(b"\xff\xd8\xff" + str(i).encode())
            frames.append(str(f))
        result = _frames_to_base64(frames)
        assert len(result) == 3

    def test_nonexistent_frame(self, tmp_path):
        """Test with a non-existent frame file (should be skipped)."""
        result = _frames_to_base64([str(tmp_path / "nonexistent.jpg")])
        assert result == []


class TestExtractSampleFrames:
    """Test frame extraction from video."""

    def test_extract_from_nonexistent_video(self):
        """Test extraction from a nonexistent file."""
        frames = _extract_sample_frames("/nonexistent/video.mp4", 10.0)
        assert frames == []

    @pytest.mark.integration
    def test_extract_from_real_video(self, tmp_video_file):
        """Test extracting frames from a real video."""
        frames = _extract_sample_frames(tmp_video_file, interval_sec=1.0, max_frames=5)
        assert len(frames) > 0
        for f in frames:
            assert os.path.isfile(f)
            assert f.endswith(".jpg")


class TestOllamaIntegration:
    """Test Ollama API calls (mocked)."""

    @patch("urllib.request.urlopen")
    def test_call_ollama(self, mock_urlopen):
        """Test calling Ollama API with mock response."""
        content = '{"summary": "test", "tags": "a,b", "objects": "c", "rating": "G"}'
        mock_response = json.dumps({"message": {"content": content}})
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _call_ollama(
            "http://localhost:11434",
            "llava",
            "system prompt",
            "user prompt",
            ["base64data"],
        )
        assert "test" in result

    @patch("urllib.request.urlopen")
    def test_call_ollama_failure(self, mock_urlopen):
        """Test Ollama API call failure returns empty string."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")
        result = _call_ollama("http://localhost:11434", "llava", "system", "user", ["data"])
        assert result == ""


class TestOpenAIIntegration:
    """Test OpenAI API calls (mocked)."""

    @patch("urllib.request.urlopen")
    def test_call_openai(self, mock_urlopen):
        """Test calling OpenAI API with mock response."""
        content = '{"summary": "test video", "tags": "a,b", "objects": "c", "rating": "PG"}'
        mock_response = json.dumps({"choices": [{"message": {"content": content}}]})
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _call_openai("sk-test-key", "gpt-4o", "system", "user", ["base64data"])
        assert "test video" in result

    @patch("urllib.request.urlopen")
    def test_call_openai_failure(self, mock_urlopen):
        """Test OpenAI API call failure."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Auth failed")
        result = _call_openai("sk-bad", "gpt-4o", "sys", "usr", ["data"])
        assert result == ""


class TestCustomAPI:
    """Test custom API calls (mocked)."""

    @patch("urllib.request.urlopen")
    def test_call_custom_api(self, mock_urlopen):
        """Test calling a custom API endpoint."""
        content = '{"summary": "custom", "tags": "x", "objects": "y", "rating": "G"}'
        mock_response = json.dumps({"choices": [{"message": {"content": content}}]})
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _call_custom_api(
            "api-key", "http://localhost:8080/v1/chat", "model", "system", "user", ["data"]
        )
        assert "custom" in result

    @patch("urllib.request.urlopen")
    def test_call_custom_api_no_key(self, mock_urlopen):
        """Test custom API with no API key."""
        mock_response = json.dumps({"choices": [{"message": {"content": '{"summary":"ok"}'}}]})
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _call_custom_api("", "http://localhost:8080/v1/chat", "model", "sys", "usr", [])
        assert "ok" in result
        # Verify the request was made without Authorization header
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert "Authorization" not in req.headers

    @patch("urllib.request.urlopen")
    def test_plain_http_non_loopback_refused_by_default(self, mock_urlopen):
        """Plain HTTP to a non-loopback host is refused unless allow_http is set."""
        result = _call_custom_api(
            "key", "http://10.0.1.4:8080/v1/chat", "model", "sys", "usr", ["data"]
        )
        assert result == ""
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_plain_http_non_loopback_allowed_with_override(self, mock_urlopen):
        """allow_http=True permits plain HTTP to a LAN host."""
        content = '{"summary": "lan", "tags": "x", "objects": "y"}'
        mock_response = json.dumps({"choices": [{"message": {"content": content}}]})
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _call_custom_api(
            "key",
            "http://10.0.1.4:8080/v1/chat",
            "model",
            "sys",
            "usr",
            ["data"],
            allow_http=True,
        )
        assert "lan" in result

    @patch("urllib.request.urlopen")
    def test_https_never_needs_override(self, mock_urlopen):
        """HTTPS to any host works without allow_http."""
        content = '{"summary": "sec", "tags": "x", "objects": "y"}'
        mock_response = json.dumps({"choices": [{"message": {"content": content}}]})
        mock_resp = MagicMock()
        mock_resp.read.return_value = mock_response.encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _call_custom_api(
            "key", "https://10.0.1.4:8080/v1/chat", "model", "sys", "usr", ["data"]
        )
        assert "sec" in result
