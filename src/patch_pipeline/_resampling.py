"""Extent-preserving resampling of channel-first volumes with SimpleITK.

Shapes follow numpy axis order. This is the only module that talks to SimpleITK,
so it is also the only place where the order is reversed to ITK's (x, y, z).
"""

from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import numpy.typing as npt
import SimpleITK as sitk

from patch_pipeline._geometry import Shape

INTERPOLATORS = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "lanczos": sitk.sitkLanczosWindowedSinc,
}

# RecursiveGaussian raises on axes shorter than this
MIN_SMOOTHABLE_LENGTH = 4


def interpolator_name(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {value!r}")
    if value not in INTERPOLATORS:
        raise ValueError(f"{name} must be one of {', '.join(INTERPOLATORS)}, got {value!r}")
    return value


@contextmanager
def sitk_threads(num_threads: int | None) -> Generator[None]:
    """Temporarily set SimpleITK's process-global thread default, which overrides per-filter
    settings. With None, the global default is left untouched."""
    if num_threads is None:
        yield
        return
    previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(num_threads)
    try:
        yield
    finally:
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(previous)


def resample_channels(
    volume: npt.NDArray[Any],
    *,
    out_shape: Shape,
    interpolator: str,
    antialias: bool,
    num_threads: int | None,
    out: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.float32]:
    """Resample each channel of a (C, Z, Y, X) volume to out_shape, into float32 `out`.

    Serves both directions: the forward stage allocates its output, the inverse stage
    passes one in. The whole stage runs within one thread-count scope.
    """
    if out is None:
        out = np.empty((len(volume), *out_shape), dtype=np.float32)
    with sitk_threads(num_threads):
        for channel, channel_out in zip(volume, out, strict=True):
            image = sitk.GetImageFromArray(np.asarray(channel, dtype=np.float32))
            if antialias:
                image = smooth_for_downsampling(image, out_shape=out_shape)
            resampled = resample_image(image, out_shape=out_shape, interpolator=interpolator)
            channel_out[...] = sitk.GetArrayViewFromImage(resampled)
    return out


def resample_image(image: sitk.Image, *, out_shape: Shape, interpolator: str) -> sitk.Image:
    """Resample a unit-spacing image so both grids cover the same extent: output voxel j
    samples the input at (j + 0.5) * N_in / N_out - 0.5."""
    in_shape = _flip_axis_order(image.GetSize())
    output_spacing = [
        in_length / out_length for in_length, out_length in zip(in_shape, out_shape, strict=True)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(_flip_axis_order(out_shape))
    resampler.SetOutputSpacing(_flip_axis_order(output_spacing))
    resampler.SetOutputOrigin(_flip_axis_order([(spacing - 1) / 2 for spacing in output_spacing]))
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(INTERPOLATORS[interpolator])
    resampler.SetOutputPixelType(sitk.sitkFloat32)
    resampler.SetUseNearestNeighborExtrapolator(True)  # safety net at the half-voxel border
    resampled: sitk.Image = resampler.Execute(image)
    return resampled


def smooth_for_downsampling(image: sitk.Image, *, out_shape: Shape) -> sitk.Image:
    """Gaussian-smooth each downsampled axis with sigma = (N_in / N_out - 1) / 2 voxels.

    Axes shorter than MIN_SMOOTHABLE_LENGTH are left unsmoothed.
    """
    in_shape = _flip_axis_order(image.GetSize())
    for axis, (in_length, out_length) in enumerate(zip(in_shape, out_shape, strict=True)):
        sigma = (in_length / out_length - 1) / 2
        if sigma > 0 and in_length >= MIN_SMOOTHABLE_LENGTH:
            image = sitk.RecursiveGaussian(image, sigma=sigma, direction=2 - axis)
    return image


def _flip_axis_order[T](values: Sequence[T]) -> list[T]:
    """Convert between numpy (z, y, x) and ITK (x, y, z) order, in either direction."""
    return list(reversed(values))
