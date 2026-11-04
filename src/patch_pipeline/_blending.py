"""Hann weighting of patch outputs and the separable normalizer that undoes it.

The normalizer is exact only because the window is separable and the patch starts form
a Cartesian product, so the total weight at a voxel is a product of three 1D sums.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from patch_pipeline._geometry import PerAxis, Shape

type Normalizers = PerAxis[npt.NDArray[np.float64]]


def hann_window(size: int) -> npt.NDArray[np.float32]:
    """Hann window that never reaches zero: the standard one of size + 2 without its endpoints."""
    return np.hanning(size + 2)[1:-1].astype(np.float32)


def patch_window(patch_size: Shape) -> npt.NDArray[np.float32]:
    window_z, window_y, window_x = (hann_window(size) for size in patch_size)
    return window_z[:, None, None] * window_y[None, :, None] * window_x[None, None, :]


def grid_normalizers(
    *, padded_shape: Shape, patch_size: Shape, starts: PerAxis[Sequence[int]]
) -> Normalizers:
    normalizer_z, normalizer_y, normalizer_x = (
        axis_normalizer(length=length, starts=axis_starts, window=hann_window(size))
        for length, size, axis_starts in zip(padded_shape, patch_size, starts, strict=True)
    )
    return normalizer_z, normalizer_y, normalizer_x


def axis_normalizer(
    *, length: int, starts: Sequence[int], window: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """Total window weight at each position along one axis, summed over the patches on it."""
    normalizer = np.zeros(length, dtype=np.float64)
    for start in starts:
        normalizer[start : start + len(window)] += window
    assert np.all(normalizer > 0), "patch starts leave positions uncovered"
    return normalizer


def normalize_in_place(accumulator: npt.NDArray[np.float32], *, normalizers: Normalizers) -> None:
    """Divide a (C, Z, Y, X) accumulator by the total window weight at each voxel."""
    normalizer_z, normalizer_y, normalizer_x = normalizers
    # a single pass over the accumulator scales each z-slice by the reciprocal of its weight
    # plane. dividing by each axis would take three passes, and it would also involve
    # casting the float64 normalizers to float32
    inverse_plane = 1 / np.outer(normalizer_y, normalizer_x)
    for z_index, weight_z in enumerate(normalizer_z):
        accumulator[:, z_index] *= (inverse_plane / weight_z).astype(np.float32)
