"""Region geometry and placement for a single high-resolution patch over a region of interest.

Mask and volume are related through edge coordinates, where voxel i spans [i, i + 1), so a
mask position e maps to e * N0 / M in the volume. The region goes where it covers the most
mask weight, and among tied positions where its centre is closest to the mask centroid.
"""

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from patch_pipeline._geometry import (
    PerAxis,
    Shape,
    finite_float32,
    keeps_source_spacing,
    per_axis,
    resolve_working_shape,
    round_half_up,
)

type Region = PerAxis[slice]

# relative to the total mask weight, absorbs cumulative-sum rounding on soft masks
TIE_RTOL = 1e-9


def roi_weights(value: npt.ArrayLike, *, name: str) -> npt.NDArray[np.float64]:
    """Validate a (Z, Y, X) mask and copy it into non-negative float64 weights."""
    mask = np.asarray(value)
    if not (
        mask.dtype == np.bool_
        or np.issubdtype(mask.dtype, np.integer)
        or np.issubdtype(mask.dtype, np.floating)
    ):
        raise TypeError(f"{name} must have a bool, integer or floating dtype, got {mask.dtype}")
    if mask.ndim != 3:
        raise ValueError(
            f"{name} must be 3D (Z, Y, X), got shape {mask.shape}; "
            "select one channel of a multi-channel output, e.g. working_output()[1]"
        )
    weights = mask.astype(np.float64)
    if not np.all(np.isfinite(weights) & (weights >= 0)):
        raise ValueError(f"{name} must be finite and non-negative")
    if not weights.any():
        raise ValueError(
            f"{name} has no positive voxels, so there is no region to place a patch on"
        )
    return weights


def region_geometry(
    *,
    original_shape: Shape,
    patch_size: Shape,
    source_spacing: PerAxis[float | None],
    target_spacing: PerAxis[float | None],
    spacing_rtol: PerAxis[float | None],
) -> tuple[Shape, Shape]:
    """The region of original voxels one patch spans, and the working shape it is resampled to.

    On an axis shorter than that span, the region is the whole axis, worked at the length
    PatchPipeline would give it, which is never longer than the patch.
    """
    whole_working_shape = resolve_working_shape(
        input_shape=original_shape,
        source_spacing=source_spacing,
        target_spacing=target_spacing,
        spacing_rtol=spacing_rtol,
        target_shape=(None, None, None),
    )
    extents = per_axis(
        _patch_extent(
            patch_size=patch_size[axis],
            source_spacing=source_spacing[axis],
            target_spacing=target_spacing[axis],
            spacing_rtol=spacing_rtol[axis],
        )
        for axis in range(3)
    )
    region_shape = per_axis(
        min(extent, length) for extent, length in zip(extents, original_shape, strict=True)
    )
    working_shape = per_axis(
        size if extent <= length else whole_working
        for extent, length, size, whole_working in zip(
            extents, original_shape, patch_size, whole_working_shape, strict=True
        )
    )
    return region_shape, working_shape


def window_sums(
    weights: npt.NDArray[np.float64], *, window_shape: Shape
) -> npt.NDArray[np.float64]:
    """Total weight of every window position, of shape `weights.shape - window_shape + 1`."""
    sums = weights
    for axis, length in enumerate(window_shape):
        # sliding sum along one axis, from cumulative sums with a leading zero
        cumulative = np.moveaxis(np.cumsum(sums, axis=axis), axis, 0)
        cumulative = np.insert(cumulative, 0, 0.0, axis=0)
        sums = np.moveaxis(cumulative[length:] - cumulative[:-length], 0, axis)
    return sums


def place_region(
    weights: npt.NDArray[np.float64], *, original_shape: Shape, region_shape: Shape
) -> tuple[Region, float]:
    """The region covering the most mask weight, and the fraction of the weight it covers."""
    window_shape = per_axis(
        min(max(round_half_up(region_length * mask_length / original_length), 1), mask_length)
        for region_length, mask_length, original_length in zip(
            region_shape, weights.shape, original_shape, strict=True
        )
    )
    sums = window_sums(weights, window_shape=window_shape)
    total = weights.sum()
    best = sums.max()
    tied_starts = np.argwhere(sums >= best - TIE_RTOL * total)
    centre_offsets = tied_starts + np.array(window_shape) / 2 - _centroid(weights)
    # argmin takes the first in C order among equal distances
    mask_start = tied_starts[np.argmin((centre_offsets**2).sum(axis=1))]
    region = per_axis(
        _region_slice(
            mask_centre=start + window_length / 2,
            mask_length=mask_length,
            original_length=original_length,
            region_length=region_length,
        )
        for start, window_length, mask_length, original_length, region_length in zip(
            mask_start, window_shape, weights.shape, original_shape, region_shape, strict=True
        )
    )
    return region, float(best / total)


def channel_fill(fill_value: object, *, n_channels: int) -> npt.NDArray[np.float32]:
    """One fill value per output channel, from a scalar or a per-channel sequence."""
    if isinstance(fill_value, Iterable) and not isinstance(fill_value, str):
        values = tuple(fill_value)
    else:
        values = (fill_value,) * n_channels
    if len(values) != n_channels:
        raise ValueError(
            f"fill_value must be a scalar or have one entry per output channel ({n_channels}), "
            f"got {len(values)}"
        )
    return np.array(
        [
            finite_float32(value, name=f"fill_value[{channel}]")
            for channel, value in enumerate(values)
        ],
        dtype=np.float32,
    )


def _patch_extent(
    *,
    patch_size: int,
    source_spacing: float | None,
    target_spacing: float | None,
    spacing_rtol: float | None,
) -> int:
    """Original voxels one patch spans along an axis, whatever the length of the volume."""
    # resolve_working_shape has checked that a target spacing comes with a source spacing
    if (
        target_spacing is None
        or source_spacing is None
        or keeps_source_spacing(
            source_spacing=source_spacing, target_spacing=target_spacing, spacing_rtol=spacing_rtol
        )
    ):
        return patch_size
    return max(1, round_half_up(patch_size * target_spacing / source_spacing))


def _centroid(weights: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Weighted mean position per axis, in edge coordinates."""
    total = weights.sum()
    return np.array(
        [
            np.dot(
                weights.sum(axis=tuple(other for other in range(3) if other != axis)),
                np.arange(length) + 0.5,
            )
            / total
            for axis, length in enumerate(weights.shape)
        ]
    )


def _region_slice(
    *, mask_centre: float, mask_length: int, original_length: int, region_length: int
) -> slice:
    """The region along one axis, centred on a mask position and kept inside the volume."""
    centre = mask_centre * original_length / mask_length
    start = round_half_up(centre - region_length / 2)
    start = min(max(start, 0), original_length - region_length)
    return slice(start, start + region_length)
