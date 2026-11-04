import gc
import weakref
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from patch_pipeline import PatchPipeline, RoiPipeline
from patch_pipeline import _pipeline as pipeline_module
from patch_pipeline._geometry import Shape
from patch_pipeline._roi import region_geometry


def random_volume(shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    return np.random.default_rng(seed=0).random(shape, dtype=np.float32)


def box_mask(mask_shape: Shape, box: tuple[slice, slice, slice]) -> npt.NDArray[np.bool_]:
    mask = np.zeros(mask_shape, dtype=bool)
    mask[box] = True
    return mask


# a 20 x 24 x 16 volume with a half-resolution mask, whose box spans voxels 6:12, 8:14, 4:10
VOLUME_SHAPE = (2, 20, 24, 16)
ROI_MASK = box_mask((10, 12, 8), (slice(3, 6), slice(4, 7), slice(2, 5)))
# z downsampled, y upsampled and x unchanged, so the region is (16, 4, 8)
RESAMPLED: dict[str, Any] = {
    "patch_size": 8,
    "source_spacing": 1.0,
    "target_spacing": (2.0, 0.5, None),
}


def roi_pipeline(volume: npt.NDArray[Any], **options: Any) -> RoiPipeline:
    return RoiPipeline(volume, **({"roi_mask": ROI_MASK} | RESAMPLED | options))


def complete_roi_pipeline(**options: Any) -> RoiPipeline:
    """A pipeline on the random volume, fed two output channels derived from its patch."""
    pipeline = roi_pipeline(random_volume(VOLUME_SHAPE), **options)
    pipeline.feed(0, pipeline[0] * 2.0)
    return pipeline


@pytest.fixture
def no_resampling(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("resampling should have been skipped")

    monkeypatch.setattr(pipeline_module, "resample_channels", fail)


def test_properties_match_hand_computed_geometry() -> None:
    pipeline = roi_pipeline(random_volume(VOLUME_SHAPE))
    assert pipeline.original_shape == (20, 24, 16)
    # per axis, two mask windows are equally close to the box centre, and the first is taken;
    # the y window of 2 mask voxels covers 2 of the box's 3
    assert pipeline.region == (slice(0, 16), slice(8, 12), slice(2, 10))
    assert pipeline.coverage == pytest.approx(2 / 3)
    assert pipeline.working_shape == (8, 8, 8)
    assert pipeline.effective_spacing == (2.0, 0.5, 1.0)


def test_effective_spacing_follows_the_rounded_region() -> None:
    pipeline = roi_pipeline(
        random_volume(VOLUME_SHAPE),
        source_spacing=(1.0, 0.7, None),
        target_spacing=(2.0, 0.5, None),
    )
    # y: 8 * 0.5 / 0.7 = 5.71 rounds to 6 original voxels
    assert pipeline.region[1].stop - pipeline.region[1].start == 6
    assert pipeline.effective_spacing == pytest.approx((2.0, 0.7 * 6 / 8, None))


def test_repr_shows_the_region_and_coverage() -> None:
    pipeline = roi_pipeline(random_volume(VOLUME_SHAPE))
    assert repr(pipeline) == (
        "RoiPipeline(original_shape=(20, 24, 16), region=(0:16, 8:12, 2:10), "
        "working_shape=(8, 8, 8), patch_size=(8, 8, 8), coverage=0.667, n_channels_in=2, "
        "n_channels_out=None, fed=0/1)"
    )


class TestCallerArraysAreNotRetained:
    @pytest.mark.parametrize("name", ["volume", "roi_mask"])
    def test_no_reference_is_kept(self, name: str) -> None:
        arrays: dict[str, Any] = {
            "volume": random_volume(VOLUME_SHAPE),
            "roi_mask": ROI_MASK.astype(np.float64),
        }
        reference = weakref.ref(arrays[name])
        pipeline = RoiPipeline(arrays.pop("volume"), roi_mask=arrays.pop("roi_mask"), **RESAMPLED)
        gc.collect()
        assert reference() is None
        assert pipeline[0].shape == (2, 8, 8, 8)

    @pytest.mark.usefixtures("no_resampling")
    def test_mutating_the_caller_volume_leaves_the_patch(self) -> None:
        volume = random_volume(VOLUME_SHAPE)
        pipeline = roi_pipeline(volume, patch_size=4, target_spacing=None)
        before = pipeline[0]
        volume[...] = -1
        np.testing.assert_array_equal(pipeline[0], before)


class TestDelegation:
    @pytest.mark.usefixtures("no_resampling")
    def test_without_resampling_the_identity_model_recovers_the_region(self) -> None:
        volume = random_volume(VOLUME_SHAPE)
        pipeline = roi_pipeline(volume, patch_size=(6, 8, 5), target_spacing=None)
        pipeline.feed(0, pipeline[0])

        result = pipeline.finalize()
        region = pipeline.region
        np.testing.assert_allclose(result[:, *region], volume[:, *region], rtol=1e-6)
        result[:, *region] = 0
        assert not result.any()

    @pytest.mark.parametrize(
        ("volume_shape", "roi_mask", "options"),
        [
            pytest.param(VOLUME_SHAPE, ROI_MASK, {}, id="resampled"),
            # z spans 16 voxels at 2.0 but has only 6, so it is worked at 3 and padded to 8
            pytest.param(
                (2, 6, 24, 16),
                box_mask((3, 12, 8), (slice(None), slice(4, 7), slice(2, 5))),
                {"target_spacing": (2.0, None, None), "pad_mode": "reflect"},
                id="padded",
            ),
        ],
    )
    def test_matches_a_patch_pipeline_on_the_region(
        self, volume_shape: Shape, roi_mask: npt.NDArray[np.bool_], options: dict[str, Any]
    ) -> None:
        volume = random_volume(volume_shape)
        pipeline = roi_pipeline(volume, roi_mask=roi_mask, **options)
        region_shape, working_shape = region_geometry(
            original_shape=pipeline.original_shape,
            patch_size=pipeline.patch_size,
            source_spacing=(1.0, 1.0, 1.0),
            target_spacing=(RESAMPLED | options)["target_spacing"],
            spacing_rtol=(None, None, None),
        )
        reference = PatchPipeline(
            volume[:, *pipeline.region],
            patch_size=8,
            target_shape=working_shape,
            pad_mode=options.get("pad_mode", "constant"),
        )
        assert tuple(axis.stop - axis.start for axis in pipeline.region) == region_shape
        assert pipeline.working_shape == working_shape

        np.testing.assert_array_equal(pipeline[0], reference[0])
        outputs = np.random.default_rng(seed=1).random((3, 8, 8, 8), dtype=np.float32)
        pipeline.feed(0, outputs)
        reference.feed(0, outputs)
        np.testing.assert_array_equal(pipeline.finalize_region(), reference.finalize())

    def test_padded_axis_uses_the_whole_axis(self) -> None:
        pipeline = roi_pipeline(random_volume((2, 6, 24, 16)), target_spacing=(2.0, None, None))
        assert pipeline.region[0] == slice(0, 6)
        assert pipeline.working_shape == (3, 8, 8)


class TestForwardedApi:
    @pytest.fixture
    def pipeline(self) -> RoiPipeline:
        return roi_pipeline(random_volume(VOLUME_SHAPE))

    def test_there_is_exactly_one_patch(self, pipeline: RoiPipeline) -> None:
        assert len(pipeline) == 1
        assert pipeline.grid_shape == (1, 1, 1)
        assert pipeline.locations.tolist() == [[0, 0, 0]]
        assert pipeline.n_channels_in == 2
        assert pipeline[0].shape == (2, 8, 8, 8)
        assert [index for index, _ in pipeline] == [0]

    def test_batches_yield_one_batch(self, pipeline: RoiPipeline) -> None:
        batches = list(pipeline.batches(1))
        assert len(batches) == 1
        indices, patches = batches[0]
        assert indices.tolist() == [0]
        assert patches.shape == (1, 2, 8, 8, 8)

    def test_feeding_completes_and_releases_the_patch(self, pipeline: RoiPipeline) -> None:
        pipeline.feed(0, np.zeros((3, 8, 8, 8)))
        assert pipeline.is_complete
        assert pipeline.fed.tolist() == [True]
        assert pipeline.n_channels_out == 3
        assert pipeline.working_output().shape == (3, 8, 8, 8)
        with pytest.raises(IndexError, match="already fed"):
            pipeline.feed(0, np.zeros((3, 8, 8, 8)))
        with pytest.raises(RuntimeError, match="working input was released"):
            pipeline[0]


class TestFinalizeRegion:
    def test_returns_the_region_shape(self) -> None:
        assert complete_roi_pipeline().finalize_region().shape == (2, 16, 4, 8)

    def test_writes_only_the_region_of_a_view(self) -> None:
        pipeline = complete_roi_pipeline()
        user_volume = np.full(VOLUME_SHAPE, -7.0, dtype=np.float32)
        result = pipeline.finalize_region(out=user_volume[:, *pipeline.region])

        assert np.shares_memory(result, user_volume)
        assert not np.any(user_volume[:, *pipeline.region] == -7.0)
        user_volume[:, *pipeline.region] = -7.0
        assert np.all(user_volume == -7.0)

    def test_wrong_out_shape_names_the_region_shape(self) -> None:
        pipeline = complete_roi_pipeline()
        with pytest.raises(ValueError, match=r"\(C_out, \*region_shape\) = \(2, 16, 4, 8\)"):
            pipeline.finalize_region(out=np.empty(VOLUME_SHAPE, dtype=np.float32))


class TestFinalize:
    @pytest.mark.parametrize(
        ("fill_value", "expected"),
        [
            pytest.param(0.0, [0.0, 0.0], id="zero"),
            pytest.param(2.5, [2.5, 2.5], id="scalar"),
            pytest.param((1.0, -30.0), [1.0, -30.0], id="per channel"),
        ],
    )
    def test_fill_reaches_every_voxel_outside_the_region(
        self, fill_value: float | tuple[float, ...], expected: list[float]
    ) -> None:
        pipeline = complete_roi_pipeline()
        region = pipeline.region
        expected_region = complete_roi_pipeline().finalize_region()

        result = pipeline.finalize(fill_value=fill_value)
        assert result.shape == VOLUME_SHAPE
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result[:, *region], expected_region)
        outside = np.ones(VOLUME_SHAPE[1:], dtype=bool)
        outside[region] = False
        assert np.all(result[:, outside] == np.array(expected)[:, None])

    def test_writes_into_a_given_out(self) -> None:
        out = np.full(VOLUME_SHAPE, -7.0, dtype=np.float32)
        result = complete_roi_pipeline().finalize(out=out, fill_value=0.0)
        assert result is out
        np.testing.assert_array_equal(out, complete_roi_pipeline().finalize())

    @pytest.mark.parametrize(
        ("fill_value", "out", "error"),
        [
            pytest.param(
                (1.0, 0.0, 0.0), np.empty(VOLUME_SHAPE, np.float32), ValueError, id="fill"
            ),
            pytest.param(1.0, np.empty((2, 20, 24, 15), np.float32), ValueError, id="out shape"),
        ],
    )
    def test_rejected_call_writes_nothing_and_can_be_retried(
        self, fill_value: Any, out: npt.NDArray[np.float32], error: type
    ) -> None:
        pipeline = complete_roi_pipeline()
        out[...] = -7.0
        with pytest.raises(error):
            pipeline.finalize(out=out, fill_value=fill_value)
        assert np.all(out == -7.0)
        assert pipeline.finalize().shape == VOLUME_SHAPE

    @pytest.mark.parametrize("finish", ["finalize", "finalize_region"])
    def test_finishing_before_feeding_raises_and_writes_nothing(self, finish: str) -> None:
        pipeline = roi_pipeline(random_volume(VOLUME_SHAPE))
        out = np.full(VOLUME_SHAPE, -7.0, dtype=np.float32)
        with pytest.raises(RuntimeError, match="1 of 1 patches have not been fed"):
            getattr(pipeline, finish)(out=out)
        assert np.all(out == -7.0)

    @pytest.mark.parametrize("first", ["finalize", "finalize_region"])
    @pytest.mark.parametrize("call_again", ["finalize", "finalize_region", "working_output"])
    def test_finalizing_is_one_shot(self, first: str, call_again: str) -> None:
        pipeline = complete_roi_pipeline()
        getattr(pipeline, first)()
        with pytest.raises(RuntimeError, match="finalize was already called"):
            getattr(pipeline, call_again)()


class TestValidation:
    @pytest.mark.parametrize(
        ("options", "error", "message"),
        [
            pytest.param({"roi_mask": np.zeros((4, 4, 4))}, ValueError, "no positive voxels"),
            pytest.param({"roi_mask": ROI_MASK[None]}, ValueError, "select one channel"),
            pytest.param({"source_spacing": None}, ValueError, "source_spacing\\[0\\] is needed"),
            pytest.param({"patch_size": 0}, ValueError, "patch_size\\[0\\] must be positive"),
            pytest.param({"interp_in": "cubic"}, ValueError, "interp_in"),
            pytest.param({"pad_mode": "empty"}, ValueError, "pad_mode"),
        ],
    )
    def test_invalid_arguments_raise(
        self, options: dict[str, Any], error: type, message: str
    ) -> None:
        with pytest.raises(error, match=message):
            roi_pipeline(random_volume(VOLUME_SHAPE), **options)
