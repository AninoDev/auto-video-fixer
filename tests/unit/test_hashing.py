"""Tests for perceptual-hash duplicate detection (docs/REQUIREMENTS.md R5.2).

Covers `compute_video_phash`/`hash_similarity` (core/analysis.py, backed by
the `avf_hashing` Rust extension with a pure-Python/NumPy fallback of the
identical pHash algorithm), the Rust/Python parity of that algorithm, and
`VideoAnalyzer.find_similar`/`find_duplicates`.

`TestHashSeparation` is the real correctness deliverable for R5.2: it builds
a small set of genuine near-duplicate and non-duplicate video pairs with
real ffmpeg encodes (different CRF/resolution/trim, and distinct lavfi
sources) and asserts the chosen algorithm + `similarity_threshold` default
actually separates them -- not just that hashing runs without crashing.
"""

import subprocess
import tempfile
from pathlib import Path

import pytest

from autovideofixer.config import Config
from autovideofixer.core.analysis import (
    VideoAnalyzer,
    _compute_phash_rs_native,
    _compute_video_phash_python,
    _dct_matrix,
    _phash_frame_python,
    compute_video_phash,
    hash_similarity,
)


def _encode(tmp_path: Path, name: str, args: list[str]) -> str:
    """ffmpeg-encode a clip with the given extra args, returning its path."""
    out = tmp_path / name
    subprocess.run(
        ["ffmpeg", "-y", *args, "-loglevel", "error", str(out)],
        capture_output=True,
        check=True,
    )
    return str(out)


class TestPerceptualHashing:
    """Basic unit tests for compute_video_phash/hash_similarity."""

    def test_hash_nonexistent_video(self):
        assert compute_video_phash("/nonexistent/video.mp4") == ""

    @pytest.mark.integration
    def test_hash_real_video_is_hex(self, tmp_video_file):
        h = compute_video_phash(tmp_video_file)
        assert len(h) == 16
        int(h, 16)  # doesn't raise

    @pytest.mark.integration
    def test_same_video_same_hash(self, tmp_video_file):
        assert compute_video_phash(tmp_video_file) == compute_video_phash(tmp_video_file)

    @pytest.mark.integration
    def test_different_content_different_hash(self, tmp_video_file, tmp_path):
        video2 = _encode(
            tmp_path,
            "different.mp4",
            ["-f", "lavfi", "-i", "smptebars=duration=1:size=320x240:rate=24", "-c:v", "libx264"],
        )
        assert compute_video_phash(tmp_video_file) != compute_video_phash(video2)

    def test_hash_similarity_identical(self):
        h = "ff00ff00ff00ff00"
        assert hash_similarity(h, h) == 1.0

    def test_hash_similarity_completely_opposite(self):
        assert hash_similarity("0000000000000000", "ffffffffffffffff") == 0.0

    def test_hash_similarity_partial(self):
        # 0xf0 vs 0xff per byte: 4 of 8 bits differ per byte -> 32/64 total.
        h1 = "f0f0f0f0f0f0f0f0"
        h2 = "ffffffffffffffff"
        assert hash_similarity(h1, h2) == pytest.approx(0.5)

    def test_hash_similarity_different_lengths(self):
        assert hash_similarity("ff00", "ff00ff00") == 0.0

    def test_hash_similarity_empty(self):
        assert hash_similarity("", "ff00ff00ff00ff00") == 0.0
        assert hash_similarity("ff00ff00ff00ff00", "") == 0.0

    def test_hash_similarity_malformed_hex_is_zero_not_raise(self):
        # Non-hex input of matching length must degrade gracefully, not raise.
        assert hash_similarity("zzzzzzzzzzzzzzzz", "ff00ff00ff00ff00") == 0.0


class TestRustPythonHashParity:
    """The Rust extension and the pure-Python fallback implement the
    identical pHash *algorithm* (DCT matrix, low-frequency threshold,
    majority-vote combination -- see the module docstring in
    rust/avf_hashing/src/lib.rs), but do not decode frames identically:
    the Rust path uses an ffmpeg `select` filter for frame-accurate decode
    + ffmpeg's own bilinear scale, while the Python fallback seeks via
    `cv2.VideoCapture.set(POS_FRAMES)` + `cv2.resize` -- different decoders
    landing on slightly different keyframe-adjacent frames and using a
    different (if similarly-named) resize implementation. That's enough to
    flip a handful of the 64 low-frequency DCT bits near their threshold,
    so bit-for-bit equality isn't the right bar (unlike avf_scenes's
    frame-differencing parity test, which decodes every frame in both
    paths and has no such sampling-alignment ambiguity). Instead, both
    paths hashing the *same* video must still classify it as a
    near-duplicate of itself under the project's own similarity threshold
    -- i.e. they'd never disagree about whether a video is a duplicate of
    itself just because one code path built it."""

    @pytest.mark.integration
    @pytest.mark.skipif(
        _compute_phash_rs_native is None, reason="avf_hashing Rust extension not built"
    )
    def test_parity_on_real_video(self, tmp_video_file):
        rust_hash = compute_video_phash(tmp_video_file)
        python_hash = _compute_video_phash_python(tmp_video_file)
        sim = hash_similarity(rust_hash, python_hash)
        assert sim >= 0.85, f"rust vs python hash of the same video too dissimilar: {sim}"

    def test_dct_matrix_is_orthonormal(self):
        import numpy as np

        d = _dct_matrix(32)
        product = d @ d.T
        assert np.allclose(product, np.eye(32), atol=1e-9)

    def test_phash_frame_deterministic(self):
        import numpy as np

        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(32, 32)).astype(np.uint8)
        dct = _dct_matrix(32)
        dct_t = dct.T
        h1 = _phash_frame_python(frame, dct, dct_t)
        h2 = _phash_frame_python(frame, dct, dct_t)
        assert h1 == h2


class TestDuplicateDetection:
    """VideoAnalyzer.find_similar()/find_duplicates() over compute_video_phash."""

    def setup_method(self):
        self.config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        self.analyzer = VideoAnalyzer(self.config)

    @pytest.mark.integration
    def test_find_similar_same_video_is_excluded(self, tmp_video_file):
        results = self.analyzer.find_similar(tmp_video_file, [tmp_video_file])
        assert tmp_video_file not in [r[0] for r in results]

    @pytest.mark.integration
    def test_find_similar_different_videos_below_threshold(self, tmp_video_file, tmp_path):
        video2 = _encode(
            tmp_path,
            "different.mp4",
            ["-f", "lavfi", "-i", "smptebars=duration=1:size=320x240:rate=24", "-c:v", "libx264"],
        )
        results = self.analyzer.find_similar(tmp_video_file, [video2], threshold=0.85)
        assert len(results) == 0

    @pytest.mark.integration
    def test_find_duplicates_batch(self, tmp_video_file, tmp_path):
        video_copy = _encode(
            tmp_path,
            "copy.mp4",
            ["-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=24", "-c:v", "libx264"],
        )
        video_diff = _encode(
            tmp_path,
            "different.mp4",
            ["-f", "lavfi", "-i", "smptebars=duration=1:size=320x240:rate=24", "-c:v", "libx264"],
        )
        results = self.analyzer.find_duplicates(
            [tmp_video_file, video_copy, video_diff], threshold=0.5
        )
        assert len(results) >= 1


@pytest.mark.integration
@pytest.mark.slow
class TestHashSeparation:
    """The R5.2 correctness deliverable: real near-duplicate vs. non-duplicate
    pairs, built with ffmpeg, must actually separate under the chosen
    algorithm + default `similarity_threshold` (0.85 -- see config.py).

    Measured scores as of this writing (pHash, 30 sampled frames,
    3s/320x240/24fps sources; see the module docstring in
    rust/avf_hashing/src/lib.rs for the algorithm):

    Near-duplicate pairs (same source, different encode/resolution/trim):
        base vs crf32 (bitrate change):        1.0000
        base vs 160x120 (resolution change):   0.9844
        base vs trimmed (0.3s off the start):  0.9531
        160x120 vs trimmed (compounded):       0.9375

    Non-duplicate pairs (distinct lavfi sources, including cross-comparing
    each source's own near-duplicate variants):
        testsrc2 vs smptebars:                 0.4375
        testsrc2 vs solid red:                 0.3594
        testsrc2 vs mandelbrot:                0.5781
        smptebars vs solid red:                0.5156
        smptebars vs mandelbrot:               0.4844
        solid red vs mandelbrot:               0.5312

    All near-duplicate pairs scored >= 0.9375; all non-duplicate pairs
    scored <= 0.5781 -- a wide margin around the 0.85 default threshold in
    both directions.
    """

    @staticmethod
    @pytest.fixture(scope="class")
    def fixture_videos(tmp_path_factory):
        tmp_path = tmp_path_factory.mktemp("hash_separation")

        def enc(name: str, lavfi: str, extra: list[str] | None = None) -> str:
            return _encode(
                tmp_path,
                name,
                ["-f", "lavfi", "-i", lavfi, "-c:v", "libx264", *(extra or [])],
            )

        videos: dict[str, str] = {}
        videos["a_base"] = enc(
            "a_base.mp4", "testsrc2=duration=3:size=320x240:rate=24", ["-crf", "18"]
        )
        videos["a_crf32"] = enc(
            "a_crf32.mp4", "testsrc2=duration=3:size=320x240:rate=24", ["-crf", "32"]
        )
        videos["a_res"] = enc(
            "a_res.mp4",
            "testsrc2=duration=3:size=320x240:rate=24",
            ["-crf", "18", "-s", "160x120"],
        )
        # Trim ~0.3s off the start of a_base, re-encoded.
        trimmed = tmp_path / "a_trim.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                "0.3",
                "-i",
                videos["a_base"],
                "-t",
                "2.3",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-loglevel",
                "error",
                str(trimmed),
            ],
            capture_output=True,
            check=True,
        )
        videos["a_trim"] = str(trimmed)

        videos["b_base"] = enc(
            "b_base.mp4", "smptebars=duration=3:size=320x240:rate=24", ["-crf", "18"]
        )
        videos["c_solid"] = enc(
            "c_solid.mp4", "color=c=red:duration=3:size=320x240:rate=24", ["-crf", "18"]
        )
        videos["d_mandel"] = enc(
            "d_mandel.mp4", "mandelbrot=size=320x240:rate=24", ["-t", "3", "-crf", "18"]
        )
        return videos

    @staticmethod
    @pytest.fixture(scope="class")
    def fixture_hashes(fixture_videos):
        return {name: compute_video_phash(path) for name, path in fixture_videos.items()}

    def test_near_duplicates_score_above_threshold(self, fixture_hashes):
        threshold = 0.85
        near_dup_pairs = [
            ("a_base", "a_crf32"),
            ("a_base", "a_res"),
            ("a_base", "a_trim"),
            ("a_res", "a_trim"),
        ]
        for x, y in near_dup_pairs:
            sim = hash_similarity(fixture_hashes[x], fixture_hashes[y])
            assert sim >= threshold, f"expected near-duplicate {x} vs {y} >= {threshold}, got {sim}"

    def test_non_duplicates_score_below_threshold(self, fixture_hashes):
        threshold = 0.85
        distinct_sources = ["a_base", "b_base", "c_solid", "d_mandel"]
        for i, x in enumerate(distinct_sources):
            for y in distinct_sources[i + 1 :]:
                sim = hash_similarity(fixture_hashes[x], fixture_hashes[y])
                msg = f"expected non-duplicate {x} vs {y} < {threshold}, got {sim}"
                assert sim < threshold, msg

    def test_variant_of_a_is_far_from_variant_of_b(self, fixture_hashes):
        """A's near-duplicate variants must stay far from B's, not just from
        B's own base -- guards against a hash that's coincidentally close to
        *any* other video regardless of content."""
        threshold = 0.85
        for a_variant in ("a_base", "a_crf32", "a_res", "a_trim"):
            sim = hash_similarity(fixture_hashes[a_variant], fixture_hashes["b_base"])
            assert sim < threshold

    def test_find_duplicates_clusters_near_duplicates_together(self, fixture_videos):
        config = Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")
        analyzer = VideoAnalyzer(config)
        paths = list(fixture_videos.values())
        clusters = analyzer.find_duplicates(paths, threshold=0.85)

        a_group = {
            fixture_videos["a_base"],
            fixture_videos["a_crf32"],
            fixture_videos["a_res"],
            fixture_videos["a_trim"],
        }
        matching = [c for c in clusters if set(c) & a_group]
        assert matching, "expected the a_* near-duplicates to form a cluster"
        assert set(matching[0]) == a_group
