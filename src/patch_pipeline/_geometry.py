"""Working-domain and patch-grid arithmetic, on plain integers and tuples.

All per-axis values follow numpy axis order. Parsing functions turn raw user
arguments into validated per-axis tuples; resolution functions only take parsed tuples.
"""

import math
import numbers
from collections.abc import Iterable
from itertools import product
from typing import Protocol

import numpy as np
import numpy.typing as npt

type PerAxis[T] = tuple[T, T, T]
type Shape = PerAxis[int]
type Padding = PerAxis[tuple[int, int]]

DEFAULT_OVERLAP = 0.5
# absorbs float error, so a spacing exactly on the spacing_rtol bound counts as within it
SPACING_RTOL_EPSILON = 1e-9
FLOAT32_MAX = float(np.finfo(np.float32).max)


class Converter[T](Protocol):
    def __call__(self, value: object, *, name: str) -> T: ...


def broadcast_per_axis[T](value: object, *, name: str, convert: Converter[T]) -> PerAxis[T]:
    """Broadcast a scalar to all axes, or check that a sequence has one entry per axis."""
    if isinstance(value, Iterable) and not isinstance(value, str):
        values = tuple(value)
    else:
        values = (value, value, value)
    if len(values) != 3:
        raise ValueError(f"{name} must be a scalar or have exactly 3 entries, got {value!r}")
    return per_axis(convert(entry, name=f"{name}[{axis}]") for axis, entry in enumerate(values))


def positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return int(value)


def positive_float(value: object, *, name: str) -> float:
    number = _real(value, name=name)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be positive and finite, got {number}")
    return number


def non_negative_float(value: object, *, name: str) -> float:
    number = _real(value, name=name)
    if not (math.isfinite(number) and number >= 0):
        raise ValueError(f"{name} must be non-negative and finite, got {number}")
    return number


def overlap_fraction(value: object, *, name: str) -> float:
    number = _real(value, name=name)
    if not 0 <= number < 1:
        raise ValueError(f"{name} must be in [0, 1), got {number}")
    return number


def finite_float32(value: object, *, name: str) -> float:
    number = _real(value, name=name)
    if not abs(number) <= FLOAT32_MAX:
        raise ValueError(f"{name} must be finite and within float32 range, got {number}")
    return number


def optional[T](convert: Converter[T]) -> Converter[T | None]:
    def convert_unless_none(value: object, *, name: str) -> T | None:
        return None if value is None else convert(value, name=name)

    return convert_unless_none


def resolve_working_shape(
    *,
    input_shape: Shape,
    source_spacing: PerAxis[float | None],
    target_spacing: PerAxis[float | None],
    spacing_rtol: PerAxis[float | None],
    target_shape: PerAxis[int | None],
) -> Shape:
    """Pick each axis' working length from its mode: target spacing, target shape or unchanged."""
    return per_axis(
        _resolve_axis_length(
            axis=axis,
            input_length=input_shape[axis],
            source_spacing=source_spacing[axis],
            target_spacing=target_spacing[axis],
            spacing_rtol=spacing_rtol[axis],
            target_shape=target_shape[axis],
        )
        for axis in range(3)
    )


def effective_spacing(
    *, input_shape: Shape, working_shape: Shape, source_spacing: PerAxis[float | None]
) -> PerAxis[float | None]:
    """The spacing the working grid actually has, on axes whose source spacing is known."""
    return per_axis(
        None if source is None else source * input_length / working
        for source, input_length, working in zip(
            source_spacing, input_shape, working_shape, strict=True
        )
    )


def resolve_stride(
    *, patch_size: Shape, stride: Shape | None, overlap: PerAxis[float] | None
) -> Shape:
    """The maximum stride per axis, from either an explicit stride or an overlap fraction."""
    if stride is not None and overlap is not None:
        raise ValueError("give either stride or overlap, not both")
    if stride is None:
        if overlap is None:
            overlap = (DEFAULT_OVERLAP, DEFAULT_OVERLAP, DEFAULT_OVERLAP)
        # flooring makes overlap a minimum; the tolerance absorbs float error
        return per_axis(
            max(1, math.floor(size * (1 - fraction) + 1e-9))
            for size, fraction in zip(patch_size, overlap, strict=True)
        )
    for axis, (step, size) in enumerate(zip(stride, patch_size, strict=True)):
        if step > size:
            raise ValueError(
                f"stride[{axis}] = {step} exceeds patch_size[{axis}] = {size}, "
                "which would leave voxels uncovered"
            )
    return stride


def undersize_padding(*, working_shape: Shape, patch_size: Shape) -> Padding:
    """Symmetric (before, after) padding that grows axes shorter than the patch to its size."""
    return per_axis(
        _split_in_halves(max(0, size - length))
        for length, size in zip(working_shape, patch_size, strict=True)
    )


def grid_starts(
    *, padded_shape: Shape, patch_size: Shape, stride: Shape
) -> PerAxis[tuple[int, ...]]:
    return per_axis(
        patch_starts(length=length, patch_size=size, stride=step)
        for length, size, step in zip(padded_shape, patch_size, stride, strict=True)
    )


def patch_starts(*, length: int, patch_size: int, stride: int) -> tuple[int, ...]:
    """Evenly spaced starts, first at 0 and last flush with the end, at most `stride` apart."""
    if length < patch_size:
        raise ValueError(f"axis length {length} is shorter than the patch size {patch_size}")
    span = length - patch_size
    n_patches = math.ceil(span / stride) + 1
    if n_patches == 1:
        return (0,)
    return tuple(round_half_up(index * span / (n_patches - 1)) for index in range(n_patches))


def round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def patch_locations(starts: PerAxis[tuple[int, ...]]) -> npt.NDArray[np.intp]:
    """Start voxel of every patch, shape (N, 3), indexed by flat patch index in C order."""
    return np.array(list(product(*starts)), dtype=np.intp)


def keeps_source_spacing(
    *, source_spacing: float, target_spacing: float, spacing_rtol: float | None
) -> bool:
    """Whether the source spacing is within `spacing_rtol` of the target, so no resampling."""
    if spacing_rtol is None:
        return False
    relative_difference = abs(source_spacing - target_spacing) / target_spacing
    return relative_difference <= spacing_rtol + SPACING_RTOL_EPSILON


def _resolve_axis_length(
    *,
    axis: int,
    input_length: int,
    source_spacing: float | None,
    target_spacing: float | None,
    spacing_rtol: float | None,
    target_shape: int | None,
) -> int:
    if target_spacing is not None and target_shape is not None:
        raise ValueError(
            f"axis {axis} has both target_spacing and target_shape; give at most one per axis"
        )
    if spacing_rtol is not None and target_spacing is None:
        raise ValueError(f"axis {axis} has a spacing_rtol, so target_spacing[{axis}] is needed")
    if target_shape is not None:
        return target_shape
    if target_spacing is None:
        return input_length
    if source_spacing is None:
        raise ValueError(f"axis {axis} has a target_spacing, so source_spacing[{axis}] is needed")
    if keeps_source_spacing(
        source_spacing=source_spacing, target_spacing=target_spacing, spacing_rtol=spacing_rtol
    ):
        return input_length
    return max(1, round_half_up(input_length * source_spacing / target_spacing))


def _real(value: object, *, name: str) -> float:
    if not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {value!r}")
    return float(value)


def _split_in_halves(total: int) -> tuple[int, int]:
    before = total // 2
    return before, total - before


def per_axis[T](values: Iterable[T]) -> PerAxis[T]:
    z, y, x = values
    return z, y, x
