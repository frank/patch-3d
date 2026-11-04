from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import SimpleITK as sitk
from scipy import ndimage

from patch_pipeline._geometry import Shape
from patch_pipeline._resampling import (
    INTERPOLATORS,
    interpolator_name,
    resample_channels,
    sitk_threads,
    smooth_for_downsampling,
)

IN_SHAPE = (12, 17, 9)


def resample(
    volume: npt.NDArray[Any], *, out_shape: Shape, interpolator: str = "linear", **options: Any
) -> npt.NDArray[np.float32]:
    options = {"antialias": False, "num_threads": None} | options
    return resample_channels(volume, out_shape=out_shape, interpolator=interpolator, **options)


def smooth(volume: npt.NDArray[np.float32], *, out_shape: Shape) -> npt.NDArray[np.float32]:
    image = smooth_for_downsampling(sitk.GetImageFromArray(volume), out_shape=out_shape)
    return sitk.GetArrayFromImage(image)


def random_volume(shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    return np.random.default_rng(seed=0).random(shape, dtype=np.float32)


def smooth_field(shape: Shape) -> npt.NDArray[np.float32]:
    z, y, x = np.meshgrid(*(np.linspace(0, 1, length) for length in shape), indexing="ij")
    field = np.sin(2 * np.pi * z) * np.cos(np.pi * y) + np.sin(3 * x + 1)
    return field[None].astype(np.float32)


class TestInterpolatorName:
    @pytest.mark.parametrize("name", list(INTERPOLATORS))
    def test_known_names_pass_through(self, name: str) -> None:
        assert interpolator_name(name, name="interp_in") == name

    def test_unknown_name_lists_the_options(self) -> None:
        with pytest.raises(ValueError, match="interp_out must be one of nearest, linear"):
            interpolator_name("cubic", name="interp_out")

    def test_non_string_raises(self) -> None:
        with pytest.raises(TypeError, match="interp_in must be a string"):
            interpolator_name(sitk.sitkLinear, name="interp_in")


@pytest.mark.parametrize(
    "out_shape",
    [
        (6, 8, 4),  # down
        (25, 40, 20),  # up
        (20, 9, 5),  # mixed
        IN_SHAPE,  # identity
        (1, 17, 9),  # n -> 1
    ],
)
def test_linear_matches_scipy_zoom(out_shape: Shape) -> None:
    volume = random_volume(IN_SHAPE)
    zoom = np.divide(out_shape, IN_SHAPE)
    expected = ndimage.zoom(volume, zoom=zoom, order=1, grid_mode=True, mode="nearest")
    np.testing.assert_allclose(resample(volume[None], out_shape=out_shape)[0], expected, atol=1e-5)


def test_one_voxel_axis_is_upsampled_by_replication() -> None:
    volume = random_volume((1, 5, 5))
    resampled = resample(volume[None], out_shape=(4, 5, 5))[0]
    np.testing.assert_allclose(resampled, np.broadcast_to(volume, (4, 5, 5)), atol=1e-6)


def test_anisotropic_ramps_land_on_their_own_axes() -> None:
    """Swapping the axis order would change the output shape and every value."""
    in_shape, out_shape = (4, 16, 32), (10, 6, 3)
    ramps = [np.arange(length, dtype=np.float64) for length in in_shape]
    volume = 100 * ramps[0][:, None, None] + 10 * ramps[1][None, :, None] + ramps[2]

    # extent-preserving source positions of the output voxels, clamped at the half-voxel border
    positions = [
        np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0, n_in - 1)
        for n_in, n_out in zip(in_shape, out_shape, strict=True)
    ]
    expected = 100 * positions[0][:, None, None] + 10 * positions[1][None, :, None] + positions[2]

    resampled = resample(volume[None], out_shape=out_shape)[0]
    assert resampled.shape == out_shape
    np.testing.assert_allclose(resampled, expected, atol=1e-3)


@pytest.mark.parametrize("interpolator", ["linear", "bspline"])
@pytest.mark.parametrize(
    ("intermediate_shape", "max_relative_error"),
    [((40, 37, 30), 0.02), ((16, 13, 11), 0.04)],
    ids=["up-then-down", "down-then-up"],
)
def test_round_trip_error_on_smooth_field_is_bounded(
    interpolator: str, intermediate_shape: Shape, max_relative_error: float
) -> None:
    field = smooth_field((24, 20, 16))
    intermediate = resample(field, out_shape=intermediate_shape, interpolator=interpolator)
    restored = resample(intermediate, out_shape=(24, 20, 16), interpolator=interpolator)
    assert np.abs(restored - field).max() < max_relative_error * np.ptp(field)


class TestResampleChannels:
    def test_channels_are_resampled_independently(self) -> None:
        volume = random_volume((3, *IN_SHAPE))
        resampled = resample(volume, out_shape=(6, 8, 4))
        for channel in range(3):
            alone = resample(volume[channel : channel + 1], out_shape=(6, 8, 4))
            np.testing.assert_array_equal(resampled[channel], alone[0])

    def test_writes_into_a_supplied_output(self) -> None:
        out = np.full((2, 6, 8, 4), np.nan, dtype=np.float32)
        returned = resample(random_volume((2, *IN_SHAPE)), out_shape=(6, 8, 4), out=out)
        assert returned is out
        assert not np.isnan(out).any()

    def test_accepts_integer_and_non_contiguous_input(self) -> None:
        volume = np.arange(2 * 8 * 8 * 8, dtype=np.int16).reshape(2, 8, 8, 8)
        strided = volume[:, ::2, :, 1:]
        resampled = resample(strided, out_shape=(4, 8, 7), interpolator="nearest")
        assert resampled.dtype == np.float32
        np.testing.assert_array_equal(resampled, strided)

    def test_antialias_smooths_before_resampling(self) -> None:
        volume = random_volume((1, 32, 8, 8))
        smoothed = smooth(volume[0], out_shape=(8, 8, 8))
        expected = resample(smoothed[None], out_shape=(8, 8, 8))
        np.testing.assert_array_equal(
            resample(volume, out_shape=(8, 8, 8), antialias=True), expected
        )


class TestSmoothForDownsampling:
    def test_matches_scipy_on_downsampled_axis_and_leaves_others_untouched(self) -> None:
        volume = random_volume((64, 12, 10)) + 1  # offset keeps the relative error meaningful
        smoothed = smooth(volume, out_shape=(16, 12, 10))  # sigma = (1.5, 0, 0)
        expected = ndimage.gaussian_filter(volume.astype(np.float64), sigma=(1.5, 0, 0))
        interior = slice(8, -8)  # recursive filter borders differ from scipy's reflect mode
        np.testing.assert_allclose(smoothed[interior], expected[interior], rtol=0.01)

    def test_without_downsampled_axes_is_identity(self) -> None:
        volume = random_volume((8, 9, 10))
        np.testing.assert_array_equal(smooth(volume, out_shape=(8, 18, 10)), volume)

    def test_skips_axes_too_short_for_the_recursive_filter(self) -> None:
        volume = random_volume((3, 8, 8))
        np.testing.assert_array_equal(smooth(volume, out_shape=(1, 8, 8)), volume)


class TestThreadScope:
    @pytest.fixture
    def requested_threads(self) -> int:
        return 1 if sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() != 1 else 2

    def test_sets_the_global_default_and_restores_it(self, requested_threads: int) -> None:
        previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        with sitk_threads(requested_threads):
            assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == requested_threads
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == previous

    def test_restores_on_exception(self, requested_threads: int) -> None:
        previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        with pytest.raises(RuntimeError), sitk_threads(requested_threads):
            raise RuntimeError
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == previous

    def test_none_never_touches_the_global_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def forbidden(*_: object) -> None:
            raise AssertionError("global thread default was changed")

        monkeypatch.setattr(sitk.ProcessObject, "SetGlobalDefaultNumberOfThreads", forbidden)
        with sitk_threads(None):
            pass
        resample(random_volume((1, 4, 4, 4)), out_shape=(2, 2, 2))

    def test_resample_channels_restores_after_success_and_failure(
        self, requested_threads: int
    ) -> None:
        previous = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        volume = random_volume((2, 4, 4, 4))
        resample(volume, out_shape=(2, 2, 2), num_threads=requested_threads)
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == previous

        too_few_channels = np.empty((1, 2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            resample(
                volume, out_shape=(2, 2, 2), num_threads=requested_threads, out=too_few_channels
            )
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == previous
