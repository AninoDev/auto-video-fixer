"""Auto Video Fixer - existing-output spec-check helpers (REQUIREMENTS.md § 6.2).

Pure, dependency-light functions (no ``Config``/``Job`` import, no I/O beyond what
callers hand in as plain dicts) so they're trivially unit-testable in isolation and
importable from both ``core/pipeline.py`` (the § 6.1/6.2 decision path) and
``core/stages/upscale.py`` (which shares the orientation-aware resolution-bounds
math with this module instead of duplicating it -- see ``effective_target_bounds()``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Shared by both UpscaleStage's "already at target?" checks
# (``UpscaleStage._SKIP_SCALE_THRESHOLD``, kept as an alias onto this constant for
# backward compat) and the § 6.2 existing-output resolution spec-check: 5% linear /
# ~10% area tolerance so a few-px-short exact-aspect fit (or a crop-shaved input)
# isn't treated as a genuine mismatch/upscale need. See AGENTS.md's "Upscaling &
# Aspect Ratio" section for the original rationale.
SKIP_SCALE_THRESHOLD = 1.05

# Framerate "satisfied" tolerance (absolute fps difference), e.g. 29.97 ~= 30,
# 23.976 ~= 24. Chosen as a fixed absolute epsilon (rather than a ratio) since the
# NTSC-vs-integer framerate gap (0.1%) is tiny in absolute terms across the whole
# practical fps range this tool deals with (23.976-60ish); a flat 0.11 comfortably
# covers every common NTSC pairing without being loose enough to blur genuinely
# different target rates (e.g. 30 vs 60).
FRAMERATE_EPSILON = 0.11

# Video/audio codec name -> family normalization: § 6.2 compares the codec FAMILY,
# not the literal encoder name, since e.g. an existing file encoded with
# "libx265"/"hevc_nvenc" both satisfy a "hevc" target the same way libx264 output
# would satisfy an "h264" target regardless of which x264 build produced it.
# Unknown/unlisted codec names normalize to their own lowercased string (so an
# exact literal match still works for codecs this table doesn't know about).
_VIDEO_CODEC_FAMILIES: dict[str, str] = {
    "h264": "h264",
    "libx264": "h264",
    "libx264rgb": "h264",
    "h264_nvenc": "h264",
    "h264_vaapi": "h264",
    "h264_qsv": "h264",
    "h264_videotoolbox": "h264",
    "hevc": "hevc",
    "h265": "hevc",
    "libx265": "hevc",
    "hevc_nvenc": "hevc",
    "hevc_vaapi": "hevc",
    "hevc_qsv": "hevc",
    "hevc_videotoolbox": "hevc",
    "vp9": "vp9",
    "libvpx-vp9": "vp9",
    "vp8": "vp8",
    "libvpx": "vp8",
    "av1": "av1",
    "libaom-av1": "av1",
    "librav1e": "av1",
    "libsvtav1": "av1",
    "av1_nvenc": "av1",
}
_AUDIO_CODEC_FAMILIES: dict[str, str] = {
    "aac": "aac",
    "libfdk_aac": "aac",
    "aac_at": "aac",
    "mp3": "mp3",
    "libmp3lame": "mp3",
    "opus": "opus",
    "libopus": "opus",
    "ac3": "ac3",
    "eac3": "eac3",
    "flac": "flac",
    "vorbis": "vorbis",
    "libvorbis": "vorbis",
    "pcm_s16le": "pcm",
    "pcm_s24le": "pcm",
}


def effective_target_bounds(
    input_width: int,
    input_height: int,
    target_width: int,
    target_height: int,
    keep_aspect_ratio: bool = True,
) -> tuple[int, int]:
    """Return the target bounding box, rotated to match the input's orientation.

    Shared by ``UpscaleStage.should_run()``/``execute()``'s method selection/
    ``_calculate_target_dimensions()`` (see ``core/stages/upscale.py``, which now
    delegates its instance method of the same name to this pure function) and the
    § 6.2 existing-output resolution spec-check (``resolution_satisfies()`` below)
    -- both need to agree on what "target resolution" means for a given input's
    orientation. If ``keep_aspect_ratio`` is False, returns the target unrotated.
    """
    if not keep_aspect_ratio:
        return target_width, target_height
    if input_height > input_width:  # Portrait input
        return target_height, target_width
    elif input_width > input_height:  # Landscape input
        return target_width, target_height
    else:  # Square input
        side = min(target_width, target_height)
        return side, side


def resolution_satisfies(
    existing_width: int,
    existing_height: int,
    target_width: int,
    target_height: int,
    keep_aspect_ratio: bool = True,
    threshold: float = SKIP_SCALE_THRESHOLD,
) -> bool:
    """True if (existing_width, existing_height) already satisfies the
    orientation-aware, threshold-tolerant target bounding box -- e.g. a
    1080x1918 output satisfies a [1920, 1080] target (rotated bounds; a few px
    short of exact aspect is fine)."""
    if existing_width <= 0 or existing_height <= 0:
        # No usable resolution info -- fall back to a non-rotated, exact
        # comparison rather than asserting a mismatch outright.
        return existing_width >= target_width and existing_height >= target_height
    bound_w, bound_h = effective_target_bounds(
        existing_width, existing_height, target_width, target_height, keep_aspect_ratio
    )
    scale_needed = max(bound_w / existing_width, bound_h / existing_height)
    return scale_needed <= threshold


def _codec_family(name: str | None, table: dict[str, str]) -> str | None:
    if not name:
        return None
    return table.get(name.strip().lower(), name.strip().lower())


@dataclass
class OutputTargets:
    """Effective § 6.2 spec-check targets derived from one job's config.

    Every field left at its default (``None`` / ``True`` for
    ``keep_aspect_ratio``) means "unspecified" -- ``check_output_spec()`` never
    treats an unspecified target as a mismatch. See ``effective_output_targets()``
    for the (deliberately conservative) derivation rules.
    """

    container: str | None = None
    target_width: int | None = None
    target_height: int | None = None
    keep_aspect_ratio: bool = True
    target_framerate: float | None = None
    video_codec: str | None = None
    audio_codec: str | None = None


def effective_output_targets(
    config: Any, job: Any, stage_entries: list[Any] | None = None
) -> OutputTargets:
    """Derive the v1 § 6.2 target set from ``config``/``job``.

    ``stage_entries`` (optional): the job's resolved ``StageOrderEntry`` list
    (``Pipeline.resolve_stage_order()``'s output) when the caller already has
    it -- used only to decide whether "interpolate" would actually run (see
    ``target_framerate`` below). Falls back to plain ``job.stages`` membership
    when omitted (e.g. a direct unit test constructing a bare job-like object).

    Deliberately conservative: a target field is only populated when the
    effective config unambiguously specifies it. When in doubt, a field is left
    unspecified (``None``) rather than guessed at, since an unspecified target
    is NEVER a mismatch (see ``check_output_spec()``) -- the fuzzy case this
    protects against is treating an incidental/default config value as if the
    user had actually asked for it.

    - ``container``: only from ``general.target_format`` (NOT
      ``general.output_container``, which is always set to something and would
      make "container" look "specified" on every run even when the user never
      set a target format explicitly).
    - ``target_width``/``target_height``: only from
      ``quality.quality_target.target_resolution`` when both dimensions are
      present and non-zero.
    - ``target_framerate``: only from
      ``quality.quality_target.target_framerate`` AND only when "interpolate"
      is actually a resolved occurrence for this job (``stage_entries``, i.e.
      it would actually run) -- an unused framerate target configured for a
      run that never interpolates isn't something the existing output can be
      judged against.
    - ``video_codec``/``audio_codec``: from whatever the job's effective encode
      settings actually specify -- ``job.stage_overrides["encode"]`` (highest
      precedence; also where ``_apply_config_encoding_overrides()`` surfaces a
      preset's ``encoding`` section) over ``config["encoding"]`` directly.
      Deliberately does NOT read ``config["stages"]["encode"]``: base
      ``stages.<name>`` config never reaches ``EncodeStage.execute()``'s
      ``codec``/``audio_codec`` kwargs (only job/occurrence overrides are
      threaded into ``execute(**kwargs)``), so a ``stages.encode.codec`` value
      wouldn't actually drive the encoder -- treating it as a target would
      flag mismatches the pipeline can never resolve (endless rename/
      reprocess churn). Likewise does NOT fall back to
      ``EncodeStage.execute()``'s own hardcoded kwarg defaults
      (``"libx264"``/``"aac"``) -- those are what the stage does when nothing
      was ever configured, not something the user "specified".
    """
    targets = OutputTargets()

    target_format = config.get("general", "target_format", default=None)
    if target_format:
        targets.container = str(target_format).lstrip(".").lower()

    quality_target = config.get("quality", "quality_target", default={}) or {}
    target_resolution = quality_target.get("target_resolution")
    if (
        target_resolution
        and isinstance(target_resolution, (list, tuple))
        and len(target_resolution) == 2
        and target_resolution[0]
        and target_resolution[1]
    ):
        targets.target_width = int(target_resolution[0])
        targets.target_height = int(target_resolution[1])
        targets.keep_aspect_ratio = bool(quality_target.get("keep_aspect_ratio", True))

    target_framerate = quality_target.get("target_framerate")
    if stage_entries is not None:
        interpolate_would_run = any(e.name == "interpolate" for e in stage_entries)
    else:
        # Fallback for callers (e.g. direct unit tests) that don't pass the
        # resolved occurrence list -- approximate with plain stage-name
        # membership.
        interpolate_would_run = "interpolate" in (getattr(job, "stages", None) or [])
    if target_framerate and interpolate_would_run:
        targets.target_framerate = float(target_framerate)

    encode_overrides = (getattr(job, "stage_overrides", None) or {}).get("encode", {}) or {}
    encoding_cfg = config.get("encoding", default={}) or {}

    video_codec = encode_overrides.get("codec") or encoding_cfg.get("video_codec")
    if video_codec:
        targets.video_codec = str(video_codec)

    audio_codec = encode_overrides.get("audio_codec") or encoding_cfg.get("audio_codec")
    if audio_codec:
        targets.audio_codec = str(audio_codec)

    return targets


def check_output_spec(
    existing_info: dict[str, Any] | None, targets: OutputTargets
) -> tuple[bool, list[str]]:
    """Compare an existing output's probed info dict against effective targets.

    ``existing_info=None`` represents an unreadable/corrupt existing output
    (ffprobe raised) -- always a MISMATCH with reason ``"unreadable"``.
    ``existing_info`` is the same shape ``get_video_info()``/
    ``ProbeResult.to_info_dict()`` returns.

    Returns ``(matches, mismatch_reasons)`` -- ``mismatch_reasons`` are short,
    human-readable fragments (e.g. ``"framerate 30<60"``,
    ``"vcodec h264!=libx265"``) suitable for direct inclusion in a log line or
    the § 6.6 JSON report.
    """
    if existing_info is None:
        return False, ["unreadable"]

    reasons: list[str] = []

    if targets.container:
        existing_container = str(existing_info.get("format") or "").lower()
        # ffprobe's format_name can be a comma-separated alias list, e.g.
        # "mov,mp4,m4a,3gp,3g2,mj2" -- match against any of them.
        containers = [c.strip() for c in existing_container.split(",") if c.strip()]
        if containers and targets.container not in containers:
            reasons.append(f"container {containers[0]}!={targets.container}")

    if targets.target_width and targets.target_height:
        w, h = existing_info.get("resolution", (0, 0))
        if not resolution_satisfies(
            w, h, targets.target_width, targets.target_height, targets.keep_aspect_ratio
        ):
            reasons.append(f"resolution {w}x{h}<{targets.target_width}x{targets.target_height}")

    if targets.target_framerate:
        fr = existing_info.get("framerate", 0) or 0
        if abs(fr - targets.target_framerate) > FRAMERATE_EPSILON:
            op = "<" if fr < targets.target_framerate else "!="
            reasons.append(f"framerate {fr}{op}{targets.target_framerate}")

    if targets.video_codec:
        existing_codec = existing_info.get("video_codec") or ""
        want_fam = _codec_family(targets.video_codec, _VIDEO_CODEC_FAMILIES)
        have_fam = _codec_family(existing_codec, _VIDEO_CODEC_FAMILIES)
        if have_fam != want_fam:
            reasons.append(f"vcodec {existing_codec}!={targets.video_codec}")

    if targets.audio_codec:
        audio_codecs = existing_info.get("audio_codecs") or []
        existing_acodec = audio_codecs[0] if audio_codecs else ""
        want_fam = _codec_family(targets.audio_codec, _AUDIO_CODEC_FAMILIES)
        have_fam = _codec_family(existing_acodec, _AUDIO_CODEC_FAMILIES)
        if have_fam != want_fam:
            reasons.append(f"acodec {existing_acodec}!={targets.audio_codec}")

    return (len(reasons) == 0, reasons)
