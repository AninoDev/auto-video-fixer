"""Tests for logclean.PIICleaner (REQUIREMENTS.md § 6.7 PII-clean log variant)."""

from __future__ import annotations

from autovideofixer.logclean import PIICleaner, get_pii_cleaner, reset_pii_cleaner


class TestInputOutputNumbering:
    def test_consistent_numbering_and_extension_preservation(self):
        cleaner = PIICleaner()
        cleaner.register_input("/data/in/clip_one.mkv")
        cleaner.register_input("/data/in/clip_two.mp4")
        text = "Processing /data/in/clip_one.mkv then /data/in/clip_two.mp4"
        cleaned = cleaner.clean(text)
        assert "input_video_01.mkv" in cleaned
        assert "input_video_02.mp4" in cleaned
        assert "clip_one" not in cleaned
        assert "clip_two" not in cleaned

    def test_output_numbering_independent_of_input(self):
        cleaner = PIICleaner()
        cleaner.register_input("/data/in/a.mkv")
        cleaner.register_output("/data/out/a_enhanced.mp4")
        cleaned = cleaner.clean("/data/in/a.mkv -> /data/out/a_enhanced.mp4")
        assert "input_video_01.mkv" in cleaned
        assert "output_video_01.mp4" in cleaned

    def test_same_value_same_placeholder_on_repeat(self):
        cleaner = PIICleaner()
        cleaner.register_input("/data/in/a.mkv")
        line1 = cleaner.clean("start /data/in/a.mkv")
        line2 = cleaner.clean("finish /data/in/a.mkv")
        assert "input_video_01.mkv" in line1
        assert "input_video_01.mkv" in line2

    def test_registering_same_path_twice_does_not_bump_counter(self):
        cleaner = PIICleaner()
        cleaner.register_input("/data/in/a.mkv")
        cleaner.register_input("/data/in/a.mkv")
        cleaner.register_input("/data/in/b.mkv")
        cleaned = cleaner.clean("/data/in/a.mkv and /data/in/b.mkv")
        assert "input_video_01.mkv" in cleaned
        assert "input_video_02.mkv" in cleaned


class TestDirectoryPlaceholders:
    def test_directory_role_placeholders_exact(self):
        cleaner = PIICleaner()
        cleaner.register_directory("/data/in", "input")
        cleaner.register_directory("/data/out", "output")
        cleaner.register_directory("/home/user/.config/auto-video-fixer", "config")
        cleaned = cleaner.clean("in=/data/in out=/data/out cfg=/home/user/.config/auto-video-fixer")
        assert "/path/to/input/" in cleaned
        assert "/path/to/output/" in cleaned
        assert "/path/to/config/" in cleaned
        assert "/data/in" not in cleaned
        assert "/data/out" not in cleaned

    def test_invalid_role_rejected(self):
        cleaner = PIICleaner()
        try:
            cleaner.register_directory("/some/dir", "bogus")
            raised = False
        except ValueError:
            raised = True
        assert raised

    def test_full_path_composes_directory_and_file_placeholder(self):
        """A path like /data/in/vid.mkv must clean to
        /path/to/input/input_video_01.mkv -- directory role + file
        placeholder composing correctly."""
        cleaner = PIICleaner()
        cleaner.register_directory("/data/in", "input")
        cleaner.register_input("/data/in/vid.mkv")
        cleaned = cleaner.clean("Reading /data/in/vid.mkv now")
        assert "/path/to/input/input_video_01.mkv" in cleaned
        assert "/data/in" not in cleaned

    def test_full_path_replaced_before_basename_or_bare_directory(self):
        """Longest-match-first: the full composed path wins over a
        standalone bare-directory or bare-basename substitution that would
        otherwise also match a substring of the same text -- and the
        shorter, standalone occurrences elsewhere in the text still get
        their own (correct) placeholders."""
        cleaner = PIICleaner()
        cleaner.register_directory("/data/in", "input")
        cleaner.register_input("/data/in/vid.mkv")
        cleaned = cleaner.clean("/data/in/vid.mkv and separately /data/in and vid.mkv alone")
        # The full path -> fully composed replacement (not a mangled partial
        # substitution of only the directory or only the filename).
        assert "/path/to/input/input_video_01.mkv and separately" in cleaned
        # The standalone bare directory elsewhere in the text -> bare
        # directory placeholder.
        assert "/path/to/input/ and input_video_01.mkv alone" in cleaned
        assert "/data/in" not in cleaned


class TestEndpointSubstitution:
    def test_endpoint_host_replaced_route_preserved(self):
        cleaner = PIICleaner()
        cleaner.register_endpoint("http://10.0.0.5:11434/api/generate")
        cleaned = cleaner.clean("Calling http://10.0.0.5:11434/api/generate now")
        assert "http://vlm-endpoint/api/generate" in cleaned
        assert "10.0.0.5" not in cleaned

    def test_multiple_endpoints_both_replaced(self):
        cleaner = PIICleaner()
        cleaner.register_endpoint("http://vlm.local:8080/v1/chat")
        cleaner.register_endpoint("https://llm.example.com/api")
        cleaned = cleaner.clean("vlm=http://vlm.local:8080/v1/chat llm=https://llm.example.com/api")
        assert "vlm.local" not in cleaned
        assert "llm.example.com" not in cleaned
        assert "/v1/chat" in cleaned
        assert "/api" in cleaned

    def test_empty_url_noop(self):
        cleaner = PIICleaner()
        cleaner.register_endpoint("")
        cleaner.register_endpoint(None)
        cleaned = cleaner.clean("nothing to see")
        assert cleaned == "nothing to see"


class TestTitleSubstitution:
    def test_title_numbered_and_substituted(self):
        cleaner = PIICleaner()
        cleaner.register_title("My Summer Vacation 2019")
        cleaned = cleaner.clean('Embedded title: "My Summer Vacation 2019"')
        assert "video_title_01" in cleaned
        assert "My Summer Vacation 2019" not in cleaned

    def test_multiple_titles_numbered_by_first_appearance(self):
        cleaner = PIICleaner()
        cleaner.register_title("First Title")
        cleaner.register_title("Second Title")
        cleaned = cleaner.clean("First Title then Second Title")
        assert "video_title_01" in cleaned
        assert "video_title_02" in cleaned


class TestTempPathsUntouched:
    def test_tmp_avf_paths_never_registered_stay_untouched(self):
        """/tmp/avf_* temp paths are never registered by any code path
        (no PII by construction) -- assert clean() leaves them as-is even
        though other real values ARE registered in the same run."""
        cleaner = PIICleaner()
        cleaner.register_input("/data/in/vid.mkv")
        cleaner.register_directory("/data/in", "input")
        text = "temp file at /tmp/avf_abc123.mkv while processing /data/in/vid.mkv"
        cleaned = cleaner.clean(text)
        assert "/tmp/avf_abc123.mkv" in cleaned
        assert "/path/to/input/input_video_01.mkv" in cleaned


class TestSingleton:
    def test_get_pii_cleaner_returns_same_instance(self):
        reset_pii_cleaner()
        a = get_pii_cleaner()
        b = get_pii_cleaner()
        assert a is b

    def test_reset_pii_cleaner_gives_fresh_instance(self):
        reset_pii_cleaner()
        cleaner = get_pii_cleaner()
        cleaner.register_input("/data/in/vid.mkv")
        reset_pii_cleaner()
        fresh = get_pii_cleaner()
        assert fresh is not cleaner
        cleaned = fresh.clean("/data/in/vid.mkv")
        assert cleaned == "/data/in/vid.mkv"  # nothing registered on the fresh instance


class TestLargeRunNumbering:
    def test_numbering_grows_past_two_digits_without_truncation(self):
        # The 2-digit numbering is a zero-padded MINIMUM width, not a cap:
        # video #100+ must keep a unique, full placeholder (input_video_100),
        # never wrap or truncate.
        cleaner = PIICleaner()
        for i in range(1, 151):
            cleaner.register_input(f"/data/in/vid{i}.mkv")
        assert cleaner.clean("vid7.mkv") == "input_video_07.mkv"
        assert cleaner.clean("vid99.mkv") == "input_video_99.mkv"
        assert cleaner.clean("vid100.mkv") == "input_video_100.mkv"
        assert cleaner.clean("vid150.mkv") == "input_video_150.mkv"
        # Consistency: the same real value keeps its placeholder on re-clean.
        assert cleaner.clean("vid100.mkv") == "input_video_100.mkv"


class TestAddJobRegistersDirectories:
    def test_add_job_registers_parent_dirs_for_gui_and_programmatic_use(self, tmp_path):
        # GUI/programmatic runs never pass through the CLI's directory
        # registration -- Pipeline.add_job() must register the parent dirs
        # itself or a clean log substitutes the filename but leaks the real
        # directory around it.
        from autovideofixer.core.pipeline import Pipeline

        reset_pii_cleaner()
        pipeline = Pipeline()
        input_file = tmp_path / "in" / "family_reunion.mp4"
        input_file.parent.mkdir()
        input_file.write_bytes(b"x")
        out_dir = tmp_path / "out"
        pipeline.add_job(str(input_file), output_path=str(out_dir / "family_reunion_enhanced.mp4"))

        cleaned = get_pii_cleaner().clean(f"processing {input_file}")
        assert "family_reunion" not in cleaned
        assert str(input_file.parent) not in cleaned
        assert "/path/to/input/input_video_01.mp4" in cleaned

        cleaned_out = get_pii_cleaner().clean(f"writing {out_dir}/family_reunion_enhanced.mp4")
        assert "family_reunion" not in cleaned_out
        assert "/path/to/output/output_video_01.mp4" in cleaned_out
        reset_pii_cleaner()
