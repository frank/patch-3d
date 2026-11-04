"""Resample a 3D volume, iterate it as overlapping patches, stitch the results back."""

from patch_pipeline._pipeline import PatchPipeline
from patch_pipeline._roi_pipeline import RoiPipeline

__version__ = "0.1.0"

__all__ = ["PatchPipeline", "RoiPipeline", "__version__"]
