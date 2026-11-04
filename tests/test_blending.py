import numpy as np
import numpy.typing as npt
import pytest

from patch_pipeline._blending import (
    axis_normalizer,
    grid_normalizers,
    hann_window,
    normalize_in_place,
    patch_window,
)
from patch_pipeline._geometry import Shape, grid_starts, patch_locations

GRIDS = [
    pytest.param((10, 13, 7), (4, 5, 3), (2, 3, 2), id="anisotropic overlap"),
    pytest.param((12, 12, 12), (4, 4, 4), (4, 4, 4), id="stride equals patch size"),
    pytest.param((9, 6, 5), (4, 6, 5), (1, 3, 5), id="unit stride and single-patch axes"),
]


def brute_force_weights(
    *, padded_shape: Shape, patch_size: Shape, stride: Shape
) -> npt.NDArray[np.float32]:
    weights = np.zeros(padded_shape, dtype=np.float32)
    window = patch_window(patch_size)
    starts = grid_starts(padded_shape=padded_shape, patch_size=patch_size, stride=stride)
    for z, y, x in patch_locations(starts):
        weights[z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]] += window
    return weights


class TestWindow:
    @pytest.mark.parametrize("size", [1, 2, 3, 8, 64])
    def test_window_has_no_zeros(self, size: int) -> None:
        window = hann_window(size)
        assert window.dtype == np.float32
        assert window.shape == (size,)
        assert np.all(window > 0)

    def test_window_is_symmetric_and_peaks_at_one(self) -> None:
        window = hann_window(7)
        np.testing.assert_array_equal(window, window[::-1])
        assert window.max() == pytest.approx(1.0)

    def test_patch_window_is_the_outer_product(self) -> None:
        window = patch_window((3, 4, 5))
        assert window.shape == (3, 4, 5)
        assert window.dtype == np.float32
        assert np.all(window > 0)
        expected = np.einsum("i,j,k->ijk", hann_window(3), hann_window(4), hann_window(5))
        np.testing.assert_allclose(window, expected, rtol=1e-6)


class TestNormalizer:
    @pytest.mark.parametrize(("padded_shape", "patch_size", "stride"), GRIDS)
    def test_separable_normalizer_matches_brute_force(
        self, padded_shape: Shape, patch_size: Shape, stride: Shape
    ) -> None:
        starts = grid_starts(padded_shape=padded_shape, patch_size=patch_size, stride=stride)
        normalizer_z, normalizer_y, normalizer_x = grid_normalizers(
            padded_shape=padded_shape, patch_size=patch_size, starts=starts
        )
        separable = np.einsum("i,j,k->ijk", normalizer_z, normalizer_y, normalizer_x)
        brute_force = brute_force_weights(
            padded_shape=padded_shape, patch_size=patch_size, stride=stride
        )
        np.testing.assert_allclose(separable, brute_force, rtol=1e-6)

    @pytest.mark.parametrize(("padded_shape", "patch_size", "stride"), GRIDS)
    def test_normalizing_weighted_constants_recovers_them(
        self, padded_shape: Shape, patch_size: Shape, stride: Shape
    ) -> None:
        channel_values = np.array([1.0, -2.5, 7.0], dtype=np.float32)
        weights = brute_force_weights(
            padded_shape=padded_shape, patch_size=patch_size, stride=stride
        )
        accumulator = channel_values[:, None, None, None] * weights
        starts = grid_starts(padded_shape=padded_shape, patch_size=patch_size, stride=stride)
        normalizers = grid_normalizers(
            padded_shape=padded_shape, patch_size=patch_size, starts=starts
        )

        normalize_in_place(accumulator, normalizers=normalizers)

        assert accumulator.dtype == np.float32
        expected = np.broadcast_to(channel_values[:, None, None, None], accumulator.shape)
        np.testing.assert_allclose(accumulator, expected, rtol=1e-6)

    def test_uncovered_positions_fail_the_assertion(self) -> None:
        with pytest.raises(AssertionError, match="uncovered"):
            axis_normalizer(length=8, starts=(0, 5), window=hann_window(3))
