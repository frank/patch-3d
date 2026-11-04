import math
from itertools import product

import numpy as np
import pytest

from patch_pipeline._geometry import (
    broadcast_per_axis,
    effective_spacing,
    grid_starts,
    keeps_source_spacing,
    non_negative_float,
    optional,
    overlap_fraction,
    patch_locations,
    patch_starts,
    positive_float,
    positive_int,
    resolve_stride,
    resolve_working_shape,
    undersize_padding,
)

UNSET = (None, None, None)


class TestBroadcastPerAxis:
    def test_scalar_broadcasts_to_all_axes(self) -> None:
        assert broadcast_per_axis(64, name="patch_size", convert=positive_int) == (64, 64, 64)

    def test_none_broadcasts_when_optional(self) -> None:
        parsed = broadcast_per_axis(None, name="target_spacing", convert=optional(positive_float))
        assert parsed == UNSET

    @pytest.mark.parametrize("value", [(1, 2, 3), [1, 2, 3], np.array([1, 2, 3])])
    def test_sequence_keeps_axis_order_as_python_ints(self, value: object) -> None:
        parsed = broadcast_per_axis(value, name="stride", convert=positive_int)
        assert parsed == (1, 2, 3)
        assert all(type(entry) is int for entry in parsed)

    @pytest.mark.parametrize("value", [(1, 2), (1, 2, 3, 4)])
    def test_wrong_length_raises(self, value: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="exactly 3 entries"):
            broadcast_per_axis(value, name="patch_size", convert=positive_int)

    def test_error_names_the_axis(self) -> None:
        with pytest.raises(ValueError, match=r"stride\[1\]"):
            broadcast_per_axis((4, 0, 4), name="stride", convert=positive_int)


@pytest.mark.parametrize(
    ("convert", "value", "error"),
    [
        (positive_int, 0, ValueError),
        (positive_int, -3, ValueError),
        (positive_int, 2.0, TypeError),
        (positive_int, "4", TypeError),
        (positive_float, 0.0, ValueError),
        (positive_float, -1.5, ValueError),
        (positive_float, math.inf, ValueError),
        (positive_float, math.nan, ValueError),
        (positive_float, "1.0", TypeError),
        (non_negative_float, -0.1, ValueError),
        (non_negative_float, math.inf, ValueError),
        (non_negative_float, None, TypeError),
        (overlap_fraction, -0.1, ValueError),
        (overlap_fraction, 1.0, ValueError),
        (overlap_fraction, None, TypeError),
    ],
)
def test_converters_reject_invalid_values(convert: object, value: object, error: type) -> None:
    with pytest.raises(error):
        broadcast_per_axis(value, name="argument", convert=convert)  # type: ignore[arg-type]


def test_converters_accept_numpy_scalars() -> None:
    assert positive_int(np.int64(3), name="n") == 3
    assert positive_float(np.float32(0.5), name="s") == 0.5
    assert non_negative_float(np.float64(0.0), name="r") == 0.0
    assert overlap_fraction(np.float64(0.25), name="o") == 0.25


class TestResolveWorkingShape:
    @pytest.mark.parametrize(
        ("input_length", "source", "target", "expected"),
        [
            (101, 1.0, 2.0, 51),  # 50.5 rounds up
            (5, 1.0, 2.0, 3),  # built-in round() would give 2
            (7, 1.0, 2.0, 4),  # built-in round() would also give 4
            (100, 1.0, 0.3, 333),
            (3, 1.0, 10.0, 1),  # never below one voxel
        ],
    )
    def test_target_spacing_rounds_half_up(
        self, input_length: int, source: float, target: float, expected: int
    ) -> None:
        working_shape = resolve_working_shape(
            input_shape=(input_length, 8, 8),
            source_spacing=(source, None, None),
            target_spacing=(target, None, None),
            spacing_rtol=UNSET,
            target_shape=UNSET,
        )
        assert working_shape == (expected, 8, 8)

    def test_shape_shape_spacing(self) -> None:
        working_shape = resolve_working_shape(
            input_shape=(512, 512, 40),
            source_spacing=(None, None, 2.5),
            target_spacing=(None, None, 0.5),
            spacing_rtol=UNSET,
            target_shape=(256, 256, None),
        )
        assert working_shape == (256, 256, 200)

    def test_mixed_with_unchanged_axis(self) -> None:
        working_shape = resolve_working_shape(
            input_shape=(30, 40, 50),
            source_spacing=(1.0, 0.7, 0.7),
            target_spacing=(2.0, None, None),
            spacing_rtol=UNSET,
            target_shape=(None, 20, None),
        )
        assert working_shape == (15, 20, 50)

    def test_all_unchanged_keeps_input_shape(self) -> None:
        working_shape = resolve_working_shape(
            input_shape=(30, 40, 50),
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=UNSET,
            spacing_rtol=UNSET,
            target_shape=UNSET,
        )
        assert working_shape == (30, 40, 50)

    def test_spacing_and_shape_on_one_axis_raises(self) -> None:
        with pytest.raises(ValueError, match="axis 1 has both"):
            resolve_working_shape(
                input_shape=(30, 40, 50),
                source_spacing=(1.0, 1.0, 1.0),
                target_spacing=(None, 2.0, None),
                spacing_rtol=UNSET,
                target_shape=(None, 20, None),
            )

    def test_target_spacing_without_source_spacing_raises(self) -> None:
        with pytest.raises(ValueError, match=r"source_spacing\[2\] is needed"):
            resolve_working_shape(
                input_shape=(30, 40, 50),
                source_spacing=(1.0, 1.0, None),
                target_spacing=(None, None, 2.0),
                spacing_rtol=UNSET,
                target_shape=UNSET,
            )


class TestSpacingRtol:
    @pytest.mark.parametrize(
        ("source", "rtol", "expected"),
        [
            (1.45, 0.05, 100),  # within tolerance, left unchanged
            # on the bounds, which float error puts beyond them without the epsilon
            (1.53, 0.02, 100),
            (1.47, 0.02, 100),
            (1.4, 0.05, 93),  # outside tolerance, resampled
            (1.5, 0.0, 100),
        ],
    )
    def test_source_spacing_within_tolerance_keeps_axis_length(
        self, source: float, rtol: float, expected: int
    ) -> None:
        working_shape = resolve_working_shape(
            input_shape=(100, 8, 8),
            source_spacing=(source, None, None),
            target_spacing=(1.5, None, None),
            spacing_rtol=(rtol, None, None),
            target_shape=UNSET,
        )
        assert working_shape == (expected, 8, 8)

    def test_spacing_rtol_without_target_spacing_raises(self) -> None:
        with pytest.raises(ValueError, match=r"target_spacing\[1\] is needed"):
            resolve_working_shape(
                input_shape=(30, 40, 50),
                source_spacing=(1.0, 1.0, 1.0),
                target_spacing=(2.0, None, None),
                spacing_rtol=(None, 0.05, None),
                target_shape=(None, 20, None),
            )

    @pytest.mark.parametrize(
        ("source", "rtol", "expected"),
        [
            (1.53, 0.02, True),  # on the bound
            (1.54, 0.02, False),
            (1.5, None, False),  # no tolerance always resamples, even at equal spacings
        ],
    )
    def test_keeps_source_spacing(self, source: float, rtol: float | None, expected: bool) -> None:
        kept = keeps_source_spacing(source_spacing=source, target_spacing=1.5, spacing_rtol=rtol)
        assert kept is expected


def test_effective_spacing_is_known_only_where_source_spacing_is() -> None:
    spacing = effective_spacing(
        input_shape=(101, 40, 50),
        working_shape=(51, 20, 50),
        source_spacing=(1.0, None, 0.8),
    )
    assert spacing == (1.0 * 101 / 51, None, 0.8)


class TestResolveStride:
    def test_default_overlap_is_half_rounded_down(self) -> None:
        stride = resolve_stride(patch_size=(128, 64, 5), stride=None, overlap=None)
        assert stride == (64, 32, 2)

    def test_overlap_is_never_below_the_request(self) -> None:
        for patch_size, overlap_percent in product(range(1, 257), range(100)):
            (stride, _, _) = resolve_stride(
                patch_size=(patch_size, 1, 1),
                stride=None,
                overlap=(overlap_percent / 100, 0.0, 0.0),
            )
            # exact integer arithmetic, free of the float error the implementation must tolerate
            assert stride == max(1, patch_size * (100 - overlap_percent) // 100)

    def test_zero_overlap_gives_patch_size(self) -> None:
        stride = resolve_stride(patch_size=(128, 64, 5), stride=None, overlap=(0.0, 0.0, 0.0))
        assert stride == (128, 64, 5)

    def test_near_total_overlap_keeps_stride_positive(self) -> None:
        stride = resolve_stride(patch_size=(10, 10, 10), stride=None, overlap=(0.99, 0.5, 0.0))
        assert stride == (1, 5, 10)

    def test_stride_equal_to_patch_size_is_accepted(self) -> None:
        stride = resolve_stride(patch_size=(8, 16, 32), stride=(8, 16, 32), overlap=None)
        assert stride == (8, 16, 32)

    def test_stride_larger_than_patch_size_raises(self) -> None:
        with pytest.raises(ValueError, match=r"stride\[2\] = 33 exceeds"):
            resolve_stride(patch_size=(8, 16, 32), stride=(8, 16, 33), overlap=None)

    def test_stride_and_overlap_together_raise(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            resolve_stride(patch_size=(8, 8, 8), stride=(4, 4, 4), overlap=(0.5, 0.5, 0.5))


def test_undersize_padding_is_symmetric_with_the_extra_voxel_after() -> None:
    padding = undersize_padding(working_shape=(3, 64, 100), patch_size=(8, 64, 64))
    assert padding == ((2, 3), (0, 0), (0, 0))


class TestPatchStarts:
    @pytest.mark.parametrize(
        ("length", "patch_size", "stride", "expected"),
        [
            (4, 4, 2, (0,)),
            (12, 4, 4, (0, 4, 8)),  # stride == patch_size, exact fit
            (10, 4, 4, (0, 3, 6)),  # stride == patch_size, evened out
            (9, 4, 2, (0, 2, 3, 5)),  # 5/3 and 10/3 round to nearest
        ],
    )
    def test_examples(
        self, length: int, patch_size: int, stride: int, expected: tuple[int, ...]
    ) -> None:
        assert patch_starts(length=length, patch_size=patch_size, stride=stride) == expected

    def test_axis_shorter_than_patch_raises(self) -> None:
        with pytest.raises(ValueError, match="shorter than the patch size"):
            patch_starts(length=3, patch_size=4, stride=2)

    def test_starts_are_flush_increasing_and_gap_bounded(self) -> None:
        for length in range(1, 41):
            for patch_size in range(1, length + 1):
                for stride in range(1, patch_size + 1):
                    starts = patch_starts(length=length, patch_size=patch_size, stride=stride)
                    gaps = np.diff(starts)
                    case = f"{length=} {patch_size=} {stride=} {starts=}"
                    assert len(starts) == math.ceil((length - patch_size) / stride) + 1, case
                    assert starts[0] == 0, case
                    assert starts[-1] == length - patch_size, case
                    assert np.all((gaps >= 1) & (gaps <= stride)), case


def test_patch_locations_follow_c_order_over_the_grid() -> None:
    starts = grid_starts(padded_shape=(8, 9, 7), patch_size=(4, 4, 4), stride=(4, 2, 3))
    grid_shape = tuple(len(axis_starts) for axis_starts in starts)
    locations = patch_locations(starts)

    assert starts == ((0, 4), (0, 2, 3, 5), (0, 3))
    assert locations.shape == (math.prod(grid_shape), 3)
    for flat_index in range(len(locations)):
        grid_index = np.unravel_index(flat_index, grid_shape)
        expected = [starts[axis][grid_index[axis]] for axis in range(3)]
        assert locations[flat_index].tolist() == expected
