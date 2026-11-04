import math
from itertools import product

import numpy as np
import numpy.typing as npt
import pytest

from patch_pipeline._geometry import PerAxis, Shape, per_axis
from patch_pipeline._roi import (
    Region,
    channel_fill,
    place_region,
    region_geometry,
    roi_weights,
    window_sums,
)

UNSET = (None, None, None)


def brute_force_window_sums(
    weights: npt.NDArray[np.float64], window_shape: Shape
) -> npt.NDArray[np.float64]:
    starts_shape = tuple(
        length - window + 1 for length, window in zip(weights.shape, window_shape, strict=True)
    )
    sums = np.empty(starts_shape)
    for start in np.ndindex(starts_shape):
        window = tuple(
            slice(first, first + length) for first, length in zip(start, window_shape, strict=True)
        )
        sums[start] = weights[window].sum()
    return sums


def mask_with_box(mask_shape: Shape, box: PerAxis[slice]) -> npt.NDArray[np.float64]:
    weights = np.zeros(mask_shape)
    weights[box] = 1.0
    return weights


def assert_region_contains(region: Region, box: PerAxis[tuple[float, float]]) -> None:
    for region_axis, (box_start, box_stop) in zip(region, box, strict=True):
        assert region_axis.start <= box_start and box_stop <= region_axis.stop


class TestRegionGeometry:
    # every case uses a patch of 128
    @pytest.mark.parametrize(
        ("original_shape", "spacings", "expected_region", "expected_working"),
        [
            # resampled axes that fit: 128 * 0.8 / 2.0 = 51.2 and 128 * 0.8 / 0.5 = 204.8
            ((300, 512, 512), ((2.0, 0.5, 0.5), (0.8,) * 3, UNSET), (51, 205, 205), (128,) * 3),
            # no target spacing, so nothing is resampled
            ((300, 512, 512), ((2.0, 0.5, 0.5), UNSET, UNSET), (128,) * 3, (128,) * 3),
            # axis 0 is kept by its spacing_rtol
            (
                (300, 512, 512),
                ((0.79, 0.5, 0.5), (0.8,) * 3, (0.02, None, None)),
                (128, 205, 205),
                (128,) * 3,
            ),
            # axis 0 is shorter than its patch extent of 51, so it is worked at 30 * 2.0 / 0.8 = 75
            ((30, 512, 512), ((2.0, 0.5, 0.5), (0.8,) * 3, UNSET), (30, 205, 205), (75, 128, 128)),
            # a factor near 1: the patch extent is 129 whatever the volume length
            ((300, 129, 30), ((1.0,) * 3, (1.01,) * 3, UNSET), (129, 129, 30), (128, 128, 30)),
        ],
    )
    def test_hand_computed_shapes(
        self,
        original_shape: Shape,
        spacings: tuple[PerAxis[float | None], PerAxis[float | None], PerAxis[float | None]],
        expected_region: Shape,
        expected_working: Shape,
    ) -> None:
        source_spacing, target_spacing, spacing_rtol = spacings
        region_shape, working_shape = region_geometry(
            original_shape=original_shape,
            patch_size=(128, 128, 128),
            source_spacing=source_spacing,
            target_spacing=target_spacing,
            spacing_rtol=spacing_rtol,
        )
        assert region_shape == expected_region
        assert working_shape == expected_working

    def test_region_fits_and_working_length_never_exceeds_patch(self) -> None:
        ratios = np.concatenate([np.linspace(0.25, 4.0, 61), 1 + np.linspace(-0.02, 0.02, 21)])
        for patch, original_length, ratio in product((1, 7, 64, 128), range(1, 300, 7), ratios):
            region_shape, working_shape = region_geometry(
                original_shape=(original_length,) * 3,
                patch_size=(patch,) * 3,
                source_spacing=(1.0,) * 3,
                target_spacing=(float(ratio),) * 3,
                spacing_rtol=UNSET,
            )
            assert region_shape[0] <= original_length
            assert working_shape[0] <= patch

    def test_target_spacing_without_source_spacing_raises(self) -> None:
        with pytest.raises(ValueError, match="source_spacing"):
            region_geometry(
                original_shape=(30, 30, 30),
                patch_size=(16, 16, 16),
                source_spacing=UNSET,
                target_spacing=(1.0, None, None),
                spacing_rtol=UNSET,
            )


@pytest.mark.parametrize("window_shape", [(1, 1, 1), (5, 6, 7), (2, 3, 4), (5, 1, 3)])
def test_window_sums_match_brute_force(window_shape: Shape) -> None:
    weights = np.random.default_rng(seed=0).random((5, 6, 7))
    sums = window_sums(weights, window_shape=window_shape)
    np.testing.assert_allclose(sums, brute_force_window_sums(weights, window_shape), rtol=1e-12)


class TestPlaceRegion:
    # mask voxels are 5 original voxels wide, so a region of 40 is a window of 8 mask voxels
    ORIGINAL_SHAPE = (100, 100, 100)
    MASK_SHAPE = (20, 20, 20)
    REGION_SHAPE = (40, 40, 40)

    def place(self, weights: npt.NDArray[np.float64]) -> tuple[Region, float]:
        return place_region(
            weights, original_shape=self.ORIGINAL_SHAPE, region_shape=self.REGION_SHAPE
        )

    def test_cube_inside_volume_is_centred(self) -> None:
        region, coverage = self.place(
            mask_with_box(self.MASK_SHAPE, (slice(6, 10), slice(8, 12), slice(5, 9)))
        )
        assert_region_contains(region, ((30, 50), (40, 60), (25, 45)))
        centres = [(axis.start + axis.stop) / 2 for axis in region]
        np.testing.assert_allclose(centres, (40, 50, 35), atol=5)
        assert coverage == 1.0

    def test_cube_at_border_is_clamped_inside_volume(self) -> None:
        region, coverage = self.place(
            mask_with_box(self.MASK_SHAPE, (slice(0, 4), slice(16, 20), slice(0, 4)))
        )
        assert region == (slice(0, 40), slice(60, 100), slice(0, 40))
        assert coverage == 1.0

    def test_roi_larger_than_patch_gets_brute_force_coverage(self) -> None:
        weights = mask_with_box(self.MASK_SHAPE, (slice(2, 16),) * 3)
        _, coverage = self.place(weights)
        brute_force_best = brute_force_window_sums(weights, (8, 8, 8)).max()
        assert coverage == brute_force_best / weights.sum() == 8**3 / 14**3

    def test_heavier_of_two_blobs_is_chosen(self) -> None:
        weights = mask_with_box(self.MASK_SHAPE, (slice(2, 6),) * 3)
        weights[14:16, 14:16, 14:16] = 1.0
        region, _ = self.place(weights)
        assert_region_contains(region, ((10, 30),) * 3)

    def test_soft_mask_lands_on_a_brute_force_maximum(self) -> None:
        weights = np.random.default_rng(seed=0).random(self.MASK_SHAPE)
        region_shape = (40, 30, 20)
        region, coverage = place_region(
            weights, original_shape=self.ORIGINAL_SHAPE, region_shape=region_shape
        )
        brute_force_sums = brute_force_window_sums(weights, (8, 6, 4))
        mask_start = tuple(axis.start // 5 for axis in region)
        np.testing.assert_allclose(brute_force_sums[mask_start], brute_force_sums.max(), rtol=1e-9)
        np.testing.assert_allclose(coverage, brute_force_sums.max() / weights.sum(), rtol=1e-9)

    @pytest.mark.parametrize(
        ("original_shape", "mask_shape", "region_shape"),
        [
            ((30, 64, 50), (7, 16, 9), (12, 20, 30)),  # mask shapes that don't divide the volume
            ((40, 40, 40), (40, 40, 40), (10, 10, 10)),  # mask on the volume grid
            ((60, 120, 90), (20, 20, 10), (30, 40, 45)),  # a different ratio per axis
        ],
    )
    def test_anisotropic_mask_contains_roi(
        self, original_shape: Shape, mask_shape: Shape, region_shape: Shape
    ) -> None:
        roi_voxel = tuple(length // 2 for length in mask_shape)
        weights = np.zeros(mask_shape)
        weights[roi_voxel] = 1.0
        region, coverage = place_region(
            weights, original_shape=original_shape, region_shape=region_shape
        )
        assert tuple(axis.stop - axis.start for axis in region) == region_shape
        assert all(
            axis.start >= 0 and axis.stop <= length
            for axis, length in zip(region, original_shape, strict=True)
        )
        scales = np.divide(original_shape, mask_shape)
        roi_extent = per_axis(
            (math.floor(voxel * scale), math.ceil((voxel + 1) * scale))
            for voxel, scale in zip(roi_voxel, scales, strict=True)
        )
        assert_region_contains(region, roi_extent)
        assert coverage == 1.0

    def test_ties_centre_region_on_centroid_where_clamping_allows(self) -> None:
        # centroid at mask (6, 10, 1), i.e. original (30, 50, 5); axis 2 is clamped at 0
        region, _ = self.place(
            mask_with_box(self.MASK_SHAPE, (slice(5, 7), slice(9, 11), slice(0, 2)))
        )
        assert region == (slice(10, 50), slice(30, 70), slice(0, 40))

    def test_placement_is_deterministic(self) -> None:
        weights = np.random.default_rng(seed=1).random(self.MASK_SHAPE) > 0.7
        first = self.place(weights.astype(np.float64))
        assert all(self.place(weights.astype(np.float64)) == first for _ in range(3))


class TestRoiWeights:
    @pytest.mark.parametrize("dtype", [np.bool_, np.uint8, np.int32, np.float32, np.float64])
    def test_accepts_bool_integer_and_floating_as_float64_copy(self, dtype: npt.DTypeLike) -> None:
        mask = np.zeros((4, 5, 6), dtype=dtype)
        mask[1, 2, 3] = 1
        weights = roi_weights(mask, name="roi_mask")
        assert weights.dtype == np.float64
        np.testing.assert_array_equal(weights, mask)
        assert not np.shares_memory(weights, mask)

    @pytest.mark.parametrize("shape", [(4, 5), (2, 4, 5, 6)])
    def test_rejects_other_dimensions_suggesting_a_channel(self, shape: tuple[int, ...]) -> None:
        with pytest.raises(ValueError, match="select one channel"):
            roi_weights(np.ones(shape), name="roi_mask")

    @pytest.mark.parametrize("mask", [np.ones((2, 2, 2), dtype=complex), np.full((2, 2, 2), "a")])
    def test_rejects_non_numeric_dtypes(self, mask: npt.NDArray[np.generic]) -> None:
        with pytest.raises(TypeError, match="dtype"):
            roi_weights(mask, name="roi_mask")

    @pytest.mark.parametrize("bad_value", [-1.0, math.nan, math.inf])
    def test_rejects_negative_and_non_finite_values(self, bad_value: float) -> None:
        mask = np.ones((2, 2, 2))
        mask[0, 1, 0] = bad_value
        with pytest.raises(ValueError, match="finite and non-negative"):
            roi_weights(mask, name="roi_mask")

    @pytest.mark.parametrize("shape", [(3, 3, 3), (0, 3, 3)])
    def test_rejects_all_zero_and_empty_masks(self, shape: Shape) -> None:
        with pytest.raises(ValueError, match="no positive voxels"):
            roi_weights(np.zeros(shape), name="roi_mask")


class TestChannelFill:
    def test_scalar_broadcasts_to_every_channel(self) -> None:
        fill = channel_fill(-5, n_channels=3)
        assert fill.dtype == np.float32
        np.testing.assert_array_equal(fill, [-5.0, -5.0, -5.0])

    @pytest.mark.parametrize("value", [(1.0, 0.0), [1, 0], np.array([1.0, 0.0])])
    def test_sequence_with_one_entry_per_channel(self, value: object) -> None:
        np.testing.assert_array_equal(channel_fill(value, n_channels=2), [1.0, 0.0])

    @pytest.mark.parametrize("value", [(1.0,), (1.0, 0.0, 0.0)])
    def test_wrong_length_raises(self, value: tuple[float, ...]) -> None:
        with pytest.raises(ValueError, match="one entry per output channel"):
            channel_fill(value, n_channels=2)

    @pytest.mark.parametrize("value", ["0", None, (0.0, "1")])
    def test_non_real_raises(self, value: object) -> None:
        with pytest.raises(TypeError, match="real number"):
            channel_fill(value, n_channels=1 if not isinstance(value, tuple) else 2)

    @pytest.mark.parametrize("value", [math.nan, -math.inf, -1e39])
    def test_non_finite_in_float32_raises(self, value: float) -> None:
        with pytest.raises(ValueError, match="float32"):
            channel_fill((0.0, value), n_channels=2)
