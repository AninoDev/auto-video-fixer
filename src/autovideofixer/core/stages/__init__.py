"""Auto Video Fixer - Core processing stages package."""

from autovideofixer.core.stages.base import BaseStage, StageResult, StageStatus, register_stage
from autovideofixer.core.stages.crop import CropStage
from autovideofixer.core.stages.deblock import DeblockStage
from autovideofixer.core.stages.denoise_video import DenoiseVideoStage
from autovideofixer.core.stages.detect import DetectStage
from autovideofixer.core.stages.downscale import DownscaleStage
from autovideofixer.core.stages.encode import EncodeStage
from autovideofixer.core.stages.hdr import HDRStage
from autovideofixer.core.stages.interpolate import InterpolateStage
from autovideofixer.core.stages.normalize_audio import NormalizeAudioStage, NormalizeVolumeStage
from autovideofixer.core.stages.remux import RemuxStage
from autovideofixer.core.stages.retime import RetimeStage
from autovideofixer.core.stages.speed import SpeedStage
from autovideofixer.core.stages.stabilize import StabilizeStage
from autovideofixer.core.stages.upscale import UpscaleStage

# Register all stages
register_stage(DetectStage)
register_stage(RetimeStage)
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

__all__ = [
    "BaseStage",
    "StageResult",
    "StageStatus",
    "DetectStage",
    "RetimeStage",
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
]
