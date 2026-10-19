"""Auto Video Fixer - Core processing stages package."""

from autovideofixer.core.stages.analysis_sidecar import AnalysisSidecarStage
from autovideofixer.core.stages.analyze import AnalyzeStage
from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus, register_stage
from autovideofixer.core.stages.crop import CropStage
from autovideofixer.core.stages.deblock import DeblockStage
from autovideofixer.core.stages.deinterlace import DeinterlaceStage
from autovideofixer.core.stages.denoise_video import DenoiseVideoStage
from autovideofixer.core.stages.detect import DetectStage
from autovideofixer.core.stages.downscale import DownscaleStage
from autovideofixer.core.stages.encode import EncodeStage
from autovideofixer.core.stages.hdr import HDRStage
from autovideofixer.core.stages.interpolate import InterpolateStage
from autovideofixer.core.stages.normalize_audio import NormalizeAudioStage, NormalizeVolumeStage
from autovideofixer.core.stages.normalize_codecs import NormalizeCodecsStage
from autovideofixer.core.stages.remux import RemuxStage
from autovideofixer.core.stages.retime import RetimeStage
from autovideofixer.core.stages.scene_subtitles import SceneSubtitlesStage
from autovideofixer.core.stages.speed import SpeedStage
from autovideofixer.core.stages.stabilize import StabilizeStage
from autovideofixer.core.stages.upscale import UpscaleStage

# Register all stages
register_stage(DetectStage)
register_stage(RetimeStage)
register_stage(NormalizeCodecsStage)
register_stage(DeinterlaceStage)
register_stage(StabilizeStage)
register_stage(CropStage)
register_stage(DownscaleStage)
register_stage(DeblockStage)
register_stage(DenoiseVideoStage)
register_stage(UpscaleStage)
register_stage(InterpolateStage)
register_stage(NormalizeVolumeStage)
register_stage(NormalizeAudioStage)
register_stage(EncodeStage)
register_stage(RemuxStage)
register_stage(SpeedStage)
register_stage(HDRStage)
register_stage(AnalyzeStage)
register_stage(AnalysisSidecarStage)
register_stage(SceneSubtitlesStage)

__all__ = [
    "BaseStage",
    "StageResult",
    "StageStatus",
    "DetectStage",
    "RetimeStage",
    "NormalizeCodecsStage",
    "DeinterlaceStage",
    "StabilizeStage",
    "CropStage",
    "DownscaleStage",
    "DeblockStage",
    "DenoiseVideoStage",
    "UpscaleStage",
    "InterpolateStage",
    "NormalizeVolumeStage",
    "NormalizeAudioStage",
    "EncodeStage",
    "RemuxStage",
    "SpeedStage",
    "HDRStage",
    "AnalyzeStage",
    "AnalysisSidecarStage",
    "SceneSubtitlesStage",
]
