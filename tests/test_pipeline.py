import gc
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import SimpleITK as sitk

from patch_pipeline import PatchPipeline
from patch_pipeline import _pipeline as pipeline_module
from patch_pipeline._resampling import resample_channels


def random_volume(shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    return np.random.default_rng(seed=0).random(shape, dtype=np.float32)


def working_input(pipeline: PatchPipeline) -> npt.NDArray[np.float32]:
    assert pipeline._working_input is not None
    return pipeline._working_input


def model_outputs(
    *, seed: int, batch_size: int | None = None, n_channels: int = 3
) -> npt.NDArray[np.float32]:
    """Random outputs for patches of 4³, batched unless batch_size is None."""
    batch_shape = () if batch_size is None else (batch_size,)
    return np.random.default_rng(seed=seed).random(
        (*batch_shape, n_channels, 4, 4, 4), dtype=np.float32
    )


def accumulator(pipeline: PatchPipeline) -> npt.NDArray[np.float32]:
    assert pipeline._accumulator is not None
    return pipeline._accumulator


@pytest.fixture
def no_resampling(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("resampling should have been skipped")

    monkeypatch.setattr(pipeline_module, "resample_channels", fail)


def test_mixed_mode_properties_match_hand_computed_geometry() -> None:
    pipeline = PatchPipeline(
        random_volume((2, 30, 40, 7)),
        source_spacing=(1.0, None, None),
        target_spacing=(2.0, None, None),
        target_shape=(None, 20, None),
        patch_size=(8, 8, 10),
        stride=(4, 8, 10),
    )

    assert pipeline.original_shape == (30, 40, 7)
    assert pipeline.working_shape == (15, 20, 7)
    assert pipeline.padding == ((0, 0), (0, 0), (1, 2))
    assert pipeline.padded_shape == (15, 20, 10)
    assert pipeline.effective_spacing == (2.0, None, None)
    assert pipeline.patch_size == (8, 8, 10)
    # z: 7 / 4 -> 3 starts at 0, 3.5, 7; y: 12 / 8 -> 3 starts at 0, 6, 12; x: one patch
    assert pipeline.grid_shape == (3, 3, 1)
    assert len(pipeline) == 9
    assert pipeline.locations[[0, 1, 3, 8]].tolist() == [
        [0, 0, 0],
        [0, 6, 0],
        [4, 0, 0],
        [7, 12, 0],
    ]
    assert pipeline.n_channels_in == 2
    assert pipeline.n_channels_out is None
    assert not pipeline.fed.any()
    assert not pipeline.is_complete
    assert working_input(pipeline).shape == (2, 15, 20, 10)
    assert working_input(pipeline).dtype == np.float32


def test_repr_shows_the_resolved_geometry() -> None:
    pipeline = PatchPipeline(random_volume((5, 6, 7)), patch_size=4)
    assert repr(pipeline) == (
        "PatchPipeline(original_shape=(5, 6, 7), working_shape=(5, 6, 7), "
        "padded_shape=(5, 6, 7), patch_size=(4, 4, 4), grid_shape=(2, 2, 3), "
        "n_channels_in=1, n_channels_out=None, fed=0/12)"
    )


def test_three_dimensional_volume_gets_a_channel_axis() -> None:
    pipeline = PatchPipeline(random_volume((5, 6, 7)), patch_size=4)
    assert pipeline.n_channels_in == 1
    assert working_input(pipeline).shape == (1, 5, 6, 7)


def test_read_only_surface_cannot_alter_internal_state() -> None:
    pipeline = PatchPipeline(random_volume((5, 6, 7)), patch_size=4)
    with pytest.raises(ValueError, match="read-only"):
        pipeline.locations[0, 0] = 1
    pipeline.fed[:] = True
    assert not pipeline.is_complete


RETENTION_CASES = [
    pytest.param({}, id="fast path"),
    pytest.param({"target_shape": (4, None, None)}, id="resampled"),
    pytest.param({"patch_size": 8}, id="padded"),
]


class TestCallerArrayIsNotRetained:
    @pytest.mark.parametrize("options", RETENTION_CASES)
    @pytest.mark.parametrize("shape", [(6, 6, 6), (2, 6, 6, 6)])
    def test_no_reference_is_kept(self, options: dict[str, Any], shape: tuple[int, ...]) -> None:
        volume = random_volume(shape)
        reference = weakref.ref(volume)
        pipeline = PatchPipeline(volume, **({"patch_size": 4} | options))
        del volume
        gc.collect()
        assert reference() is None
        assert working_input(pipeline).shape[0] == pipeline.n_channels_in

    @pytest.mark.parametrize("options", RETENTION_CASES)
    def test_mutating_the_caller_array_leaves_the_working_volume(
        self, options: dict[str, Any]
    ) -> None:
        volume = random_volume((6, 6, 6))
        pipeline = PatchPipeline(volume, **({"patch_size": 4} | options))
        before = working_input(pipeline).copy()
        volume[...] = -1
        np.testing.assert_array_equal(working_input(pipeline), before)


class TestFastPath:
    @pytest.mark.usefixtures("no_resampling")
    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({}, id="no targets"),
            pytest.param({"target_shape": (6, 7, 8)}, id="target shape equals original"),
            pytest.param({"original_shape": (12, 7, 3)}, id="preresampled volume"),
            pytest.param(
                {"source_spacing": 1.5, "target_spacing": 1.5}, id="target spacing equals source"
            ),
            pytest.param(
                {"source_spacing": 1.45, "target_spacing": 1.5, "spacing_rtol": 0.05},
                id="source spacing within spacing_rtol",
            ),
        ],
    )
    def test_unchanged_axes_skip_resampling(self, options: dict[str, Any]) -> None:
        volume = np.arange(6 * 7 * 8, dtype=np.int16).reshape(6, 7, 8)
        pipeline = PatchPipeline(volume, patch_size=4, **options)
        assert working_input(pipeline).dtype == np.float32
        np.testing.assert_array_equal(working_input(pipeline)[0], volume)

    def test_one_changed_axis_resamples(self) -> None:
        pipeline = PatchPipeline(
            random_volume((6, 7, 8)), patch_size=4, target_shape=(None, 9, None)
        )
        assert working_input(pipeline).shape == (1, 6, 9, 8)


class TestUndersizedAxes:
    def test_padding_is_reflected_in_the_shape_properties(self) -> None:
        pipeline = PatchPipeline(random_volume((3, 20, 2)), patch_size=(8, 8, 5))
        assert pipeline.working_shape == (3, 20, 2)
        assert pipeline.padding == ((2, 3), (0, 0), (1, 2))
        assert pipeline.padded_shape == (8, 20, 5)
        assert pipeline.grid_shape == (1, 4, 1)
        assert working_input(pipeline).shape == (1, 8, 20, 5)

    def test_constant_mode_pads_with_pad_value(self) -> None:
        volume = random_volume((2, 3, 4, 4))
        pipeline = PatchPipeline(volume, patch_size=(6, 4, 4), pad_value=-7.0)
        padded = working_input(pipeline)
        np.testing.assert_array_equal(padded[:, 1:4], volume)
        np.testing.assert_array_equal(padded[:, [0, 4, 5]], -7.0)

    def test_other_modes_are_passed_to_np_pad(self) -> None:
        volume = random_volume((3, 4, 4))
        pipeline = PatchPipeline(volume, patch_size=(6, 4, 4), pad_mode="edge")
        expected = np.pad(volume, pad_width=((1, 2), (0, 0), (0, 0)), mode="edge")
        np.testing.assert_array_equal(working_input(pipeline)[0], expected)


class TestValidation:
    @pytest.mark.parametrize(
        ("volume", "error", "message"),
        [
            (np.zeros((4, 4)), ValueError, "3D .* or 4D"),
            (np.zeros((1, 1, 4, 4, 4)), ValueError, "3D .* or 4D"),
            (np.zeros((0, 4, 4, 4)), ValueError, "must not be empty"),
            (np.zeros((4, 0, 4)), ValueError, "must not be empty"),
            (np.zeros((4, 4, 4), dtype=bool), TypeError, "integer or floating dtype"),
            (np.zeros((4, 4, 4), dtype=np.complex64), TypeError, "integer or floating dtype"),
            (np.full((4, 4, 4), "a"), TypeError, "integer or floating dtype"),
        ],
    )
    def test_invalid_volume_raises(self, volume: Any, error: type, message: str) -> None:
        with pytest.raises(error, match=message):
            PatchPipeline(volume, patch_size=2)

    @pytest.mark.usefixtures("no_resampling")
    @pytest.mark.parametrize(
        ("options", "error", "message"),
        [
            ({"pad_mode": "empty"}, ValueError, "pad_mode must be one of constant, edge"),
            ({"pad_mode": "zeros"}, ValueError, "pad_mode must be one of"),
            ({"pad_mode": np.pad}, TypeError, "pad_mode must be a string"),
            ({"interp_in": "cubic"}, ValueError, "interp_in must be one of"),
            ({"interp_out": "cubic"}, ValueError, "interp_out must be one of"),
            ({"num_threads": 0}, ValueError, "num_threads must be positive"),
            ({"stride": 2, "overlap": 0.5}, ValueError, "not both"),
            ({"stride": (2, 2, 5)}, ValueError, r"stride\[2\] = 5 exceeds"),
            ({"overlap": 1.0}, ValueError, r"overlap\[0\] must be in \[0, 1\)"),
            ({"patch_size": (4, 4)}, ValueError, "patch_size must be a scalar or have exactly 3"),
            ({"original_shape": (6, 6)}, ValueError, "original_shape must be a scalar or have"),
            ({"original_shape": (6, 0, 6)}, ValueError, r"original_shape\[1\] must be positive"),
            ({"original_shape": 6.0}, TypeError, r"original_shape\[0\] must be an integer"),
            (
                {"target_spacing": 2.0, "target_shape": None},
                ValueError,
                r"source_spacing\[0\] is needed",
            ),
            ({"spacing_rtol": -0.1}, ValueError, r"spacing_rtol\[0\] must be non-negative"),
            ({"spacing_rtol": 0.05}, ValueError, r"target_spacing\[0\] is needed"),
        ],
    )
    def test_invalid_parameters_raise_before_resampling(
        self, options: dict[str, Any], error: type, message: str
    ) -> None:
        # target_shape makes resampling necessary, so the no_resampling fixture catches late checks
        defaults: dict[str, Any] = {"patch_size": 4, "target_shape": (5, None, None)}
        with pytest.raises(error, match=message):
            PatchPipeline(random_volume((6, 6, 6)), **(defaults | options))


class TestFetching:
    @pytest.fixture
    def volume(self) -> npt.NDArray[np.float32]:
        # distinct voxel values, so any misplaced patch region shows up as a mismatch
        return np.arange(2 * 6 * 7 * 8, dtype=np.float32).reshape(2, 6, 7, 8)

    @pytest.fixture
    def pipeline(self, volume: npt.NDArray[np.float32]) -> PatchPipeline:
        return PatchPipeline(volume, patch_size=4)

    @staticmethod
    def expected_patch(
        pipeline: PatchPipeline, volume: npt.NDArray[np.float32], index: int
    ) -> npt.NDArray[np.float32]:
        z, y, x = pipeline.locations[index]
        depth, height, width = pipeline.patch_size
        return volume[:, z : z + depth, y : y + height, x : x + width]

    def test_integer_index_returns_one_patch(
        self, pipeline: PatchPipeline, volume: npt.NDArray[np.float32]
    ) -> None:
        for index in range(len(pipeline)):
            patch = pipeline[index]
            assert patch.shape == (2, 4, 4, 4)
            assert patch.dtype == np.float32
            np.testing.assert_array_equal(patch, self.expected_patch(pipeline, volume, index))

    def test_negative_index_counts_from_the_end(self, pipeline: PatchPipeline) -> None:
        np.testing.assert_array_equal(pipeline[-1], pipeline[len(pipeline) - 1])
        np.testing.assert_array_equal(pipeline[np.int64(-2)], pipeline[len(pipeline) - 2])

    @pytest.mark.parametrize(
        ("idx", "expected_indices"),
        [
            pytest.param(slice(1, 4), [1, 2, 3], id="slice"),
            pytest.param(slice(None, None, -5), [17, 12, 7, 2], id="reversed slice"),
            pytest.param([5, 0, -1], [5, 0, 17], id="list"),
            pytest.param(np.array([3, 3]), [3, 3], id="array with repeats"),
            pytest.param(np.arange(18) % 6 == 0, [0, 6, 12], id="boolean mask"),
            pytest.param([True] + [False] * 17, [0], id="boolean list"),
        ],
    )
    def test_batch_index_returns_stacked_patches(
        self,
        pipeline: PatchPipeline,
        volume: npt.NDArray[np.float32],
        idx: Any,
        expected_indices: list[int],
    ) -> None:
        assert len(pipeline) == 18
        patches = pipeline[idx]
        assert patches.shape == (len(expected_indices), 2, 4, 4, 4)
        assert patches.dtype == np.float32
        for patch, index in zip(patches, expected_indices, strict=True):
            np.testing.assert_array_equal(patch, self.expected_patch(pipeline, volume, index))

    @pytest.mark.parametrize("idx", [[], slice(5, 5), np.zeros(18, dtype=bool)])
    def test_empty_batch_has_a_zero_length_batch_axis(
        self, pipeline: PatchPipeline, idx: Any
    ) -> None:
        assert pipeline[idx].shape == (0, 2, 4, 4, 4)

    @pytest.mark.parametrize(
        ("idx", "message"),
        [
            pytest.param(18, "18 patches: index 18 is out of bounds", id="past the end"),
            pytest.param(-19, "index -19 is out of bounds", id="negative past the start"),
            pytest.param([0, 18], "out of bounds", id="list"),
            pytest.param(np.ones(3, dtype=bool), "is 18 but .* is 3", id="mask"),
            pytest.param(1.5, "only integers", id="float"),
            pytest.param((0, 1), "too many indices", id="grid coordinates"),
            pytest.param(None, r"scalar or one-dimensional.*\(1, 18\)", id="new axis"),
            pytest.param(True, "scalar or one-dimensional", id="boolean scalar"),
            pytest.param(np.zeros((2, 2), dtype=int), "scalar or one-dimensional", id="2D array"),
        ],
    )
    def test_invalid_index_raises(self, pipeline: PatchPipeline, idx: Any, message: str) -> None:
        with pytest.raises(IndexError, match=message):
            pipeline[idx]

    @pytest.mark.parametrize("idx", [0, [0, 1]])
    def test_fetched_patches_are_copies(self, pipeline: PatchPipeline, idx: Any) -> None:
        patches = pipeline[idx]
        assert not np.shares_memory(patches, working_input(pipeline))
        before = working_input(pipeline).copy()
        patches[...] = -1
        np.testing.assert_array_equal(working_input(pipeline), before)

    def test_iteration_yields_every_index_with_its_patch(self, pipeline: PatchPipeline) -> None:
        items = list(pipeline)
        assert [index for index, _ in items] == list(range(len(pipeline)))
        for index, patch in items:
            np.testing.assert_array_equal(patch, pipeline[index])

    @pytest.mark.parametrize(
        ("batch_size", "expected_sizes"),
        [(5, [5, 5, 5, 3]), (6, [6, 6, 6]), (1, [1] * 18), (100, [18])],
    )
    def test_batches_cover_all_indices_in_order(
        self, pipeline: PatchPipeline, batch_size: int, expected_sizes: list[int]
    ) -> None:
        batches = list(pipeline.batches(batch_size))
        assert [len(indices) for indices, _ in batches] == expected_sizes
        np.testing.assert_array_equal(
            np.concatenate([indices for indices, _ in batches]), np.arange(len(pipeline))
        )
        for indices, patches in batches:
            assert indices.dtype == np.intp
            np.testing.assert_array_equal(patches, pipeline[indices])

    @pytest.mark.parametrize(
        ("batch_size", "error"), [(0, ValueError), (-1, ValueError), (2.0, TypeError)]
    )
    def test_invalid_batch_size_raises(
        self, pipeline: PatchPipeline, batch_size: Any, error: type
    ) -> None:
        with pytest.raises(error, match="batch_size must"):
            next(pipeline.batches(batch_size))


class TestFeeding:
    @pytest.fixture
    def pipeline(self) -> PatchPipeline:
        # 18 patches of 4³ at stride 2, so neighbouring patches overlap
        return PatchPipeline(random_volume((2, 6, 7, 8)), patch_size=4)

    @pytest.fixture
    def fed_pipeline(self, pipeline: PatchPipeline) -> PatchPipeline:
        pipeline.feed(0, model_outputs(seed=0))
        return pipeline

    def test_one_patch_is_window_weighted_into_its_region(self, pipeline: PatchPipeline) -> None:
        outputs = model_outputs(seed=0)
        pipeline.feed(5, outputs)

        expected = np.zeros((3, *pipeline.padded_shape), dtype=np.float32)
        expected[:, *pipeline._patch_region(5)] = outputs * pipeline._window
        np.testing.assert_array_equal(accumulator(pipeline), expected)
        assert pipeline.n_channels_out == 3
        assert pipeline.fed.tolist() == [index == 5 for index in range(18)]

    @pytest.mark.parametrize(
        "idx",
        [
            pytest.param(np.arange(18), id="array"),
            pytest.param(slice(None), id="slice"),
            pytest.param(slice(None, None, -1), id="reversed slice"),
            pytest.param(list(range(-18, 0)), id="negative"),
            pytest.param(np.ones(18, dtype=bool), id="boolean mask"),
        ],
    )
    def test_batched_feed_equals_sequential_feed(self, idx: Any) -> None:
        outputs = model_outputs(seed=0, batch_size=18)
        batched = PatchPipeline(random_volume((2, 6, 7, 8)), patch_size=4)
        sequential = PatchPipeline(random_volume((2, 6, 7, 8)), patch_size=4)

        batched.feed(idx, outputs[idx])
        for index in range(18):
            sequential.feed(index, outputs[index])

        np.testing.assert_allclose(accumulator(batched), accumulator(sequential), rtol=1e-6)
        assert batched.is_complete

    @pytest.mark.parametrize(
        ("idx", "expected_indices"),
        [
            pytest.param(-1, [17], id="negative"),
            pytest.param(slice(2, 8, 3), [2, 5], id="slice"),
            pytest.param(np.arange(18) % 9 == 0, [0, 9], id="boolean mask"),
        ],
    )
    def test_fed_mask_marks_the_selected_patches(
        self, pipeline: PatchPipeline, idx: Any, expected_indices: list[int]
    ) -> None:
        # indexing per-patch outputs with idx gives one patch or a batch, matching the feed
        pipeline.feed(idx, model_outputs(seed=0, batch_size=18)[idx])
        assert np.flatnonzero(pipeline.fed).tolist() == expected_indices

    @pytest.mark.parametrize(
        "convert",
        [
            pytest.param(lambda outputs: outputs.astype(np.float64), id="float64"),
            pytest.param(lambda outputs: outputs.round().astype(np.int16), id="int16"),
            pytest.param(lambda outputs: outputs.tolist(), id="nested list"),
        ],
    )
    def test_outputs_are_cast_to_float32(self, pipeline: PatchPipeline, convert: Any) -> None:
        outputs = 10 * model_outputs(seed=0)
        pipeline.feed(0, convert(outputs))
        assert accumulator(pipeline).dtype == np.float32
        np.testing.assert_allclose(
            accumulator(pipeline)[:, :4, :4, :4],
            np.asarray(convert(outputs), dtype=np.float32) * pipeline._window,
            rtol=1e-6,
        )

    @pytest.mark.parametrize(
        ("idx", "outputs", "error", "message"),
        [
            *(
                pytest.param(
                    empty,
                    model_outputs(seed=1, batch_size=0),
                    IndexError,
                    "at least one patch index",
                    id=f"empty {form}",
                )
                for form, empty in [
                    ("list", []),
                    ("slice", slice(5, 5)),
                    ("mask", np.zeros(18, dtype=bool)),
                ]
            ),
            pytest.param(
                [1, 2, 1],
                model_outputs(seed=1, batch_size=3),
                IndexError,
                r"unique within a feed, got repeats of \[1\]",
                id="duplicate",
            ),
            pytest.param(
                [1, 0],
                model_outputs(seed=1, batch_size=2),
                IndexError,
                r"already fed: \[0\]",
                id="already fed",
            ),
            pytest.param(18, model_outputs(seed=1), IndexError, "out of bounds", id="out of range"),
            pytest.param(
                np.ones(3, dtype=bool),
                model_outputs(seed=1, batch_size=3),
                IndexError,
                "is 18 but .* is 3",
                id="mask length",
            ),
            pytest.param(
                1,
                model_outputs(seed=1, n_channels=2),
                ValueError,
                r"\(C_out, \*patch_size\) = \(3, 4, 4, 4\), got \(2, 4, 4, 4\)",
                id="channel mismatch",
            ),
            pytest.param(
                [1, 2],
                model_outputs(seed=1, batch_size=3),
                ValueError,
                r"\(B, C_out, \*patch_size\) = \(2, 3, 4, 4, 4\), got \(3, 3, 4, 4, 4\)",
                id="batch length",
            ),
            pytest.param(
                1,
                model_outputs(seed=1, batch_size=1),
                ValueError,
                "got \\(1, 3, 4, 4, 4\\)",
                id="batch outputs for one index",
            ),
            pytest.param(
                [1],
                model_outputs(seed=1),
                ValueError,
                r"= \(1, 3, 4, 4, 4\), got \(3, 4, 4, 4\)",
                id="one patch outputs for a batch",
            ),
            pytest.param(
                1,
                np.zeros((3, 4, 4, 5)),
                ValueError,
                r"got \(3, 4, 4, 5\)",
                id="patch size",
            ),
            pytest.param(
                1,
                np.zeros((3, 4, 4, 4), dtype=bool),
                TypeError,
                "outputs must have an integer or floating dtype",
                id="boolean dtype",
            ),
        ],
    )
    def test_rejected_feed_leaves_the_state_unchanged(
        self,
        fed_pipeline: PatchPipeline,
        idx: Any,
        outputs: npt.NDArray[Any],
        error: type,
        message: str,
    ) -> None:
        accumulator_before = accumulator(fed_pipeline).copy()
        fed_before = fed_pipeline.fed
        with pytest.raises(error, match=message):
            fed_pipeline.feed(idx, outputs)
        np.testing.assert_array_equal(accumulator(fed_pipeline), accumulator_before)
        np.testing.assert_array_equal(fed_pipeline.fed, fed_before)
        assert fed_pipeline.n_channels_out == 3

    @pytest.mark.parametrize(
        "outputs",
        [
            pytest.param(np.zeros((4, 4, 4)), id="no channel axis"),
            pytest.param(np.zeros((3, 4, 4, 5)), id="patch size"),
        ],
    )
    def test_rejected_first_feed_allocates_nothing(
        self, pipeline: PatchPipeline, outputs: npt.NDArray[Any]
    ) -> None:
        with pytest.raises(ValueError, match="outputs must have shape"):
            pipeline.feed(0, outputs)
        assert pipeline.n_channels_out is None
        assert pipeline._accumulator is None
        assert not pipeline.fed.any()

    def test_refeeding_lists_only_the_first_few_indices(self, pipeline: PatchPipeline) -> None:
        pipeline.feed(slice(None), model_outputs(seed=0, batch_size=18))
        with pytest.raises(IndexError, match=r"fed: \[0, 1, .*, 9, \.\.\.\] \(18 in total\)"):
            pipeline.feed(slice(None), model_outputs(seed=0, batch_size=18))


def feed_identity_model(pipeline: PatchPipeline, *, batch_size: int = 4) -> None:
    """Feed every patch back unchanged, as a model that returns its input would."""
    for indices, patches in pipeline.batches(batch_size):
        pipeline.feed(indices, patches)


def complete_pipeline(volume: npt.NDArray[Any], **options: Any) -> PatchPipeline:
    pipeline = PatchPipeline(volume, **({"patch_size": 4} | options))
    feed_identity_model(pipeline)
    return pipeline


class TestEndToEnd:
    @pytest.mark.parametrize("interpolator", ["nearest", "linear", "bspline"])
    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({"target_shape": (5, 8, 11)}, id="downsampled"),
            pytest.param({"target_shape": (19, 30, 41)}, id="upsampled"),
            pytest.param(
                {"source_spacing": (2.5, None, None), "target_spacing": (0.8, None, None)},
                id="mixed",
            ),
        ],
    )
    def test_constant_input_gives_constant_output_up_to_the_borders(
        self, interpolator: str, options: dict[str, Any]
    ) -> None:
        options = {"target_shape": (None, 30, 7)} | options
        pipeline = complete_pipeline(
            np.full((2, 11, 17, 23), 3.5, dtype=np.float32),
            patch_size=(6, 8, 5),
            interp_in=interpolator,
            interp_out=interpolator,
            **options,
        )
        result = pipeline.finalize()
        assert result.shape == (2, 11, 17, 23)
        np.testing.assert_allclose(result, 3.5, rtol=1e-6)

    @pytest.mark.usefixtures("no_resampling")
    @pytest.mark.parametrize(
        "spacing",
        [
            pytest.param({"overlap": 0.5}, id="overlap 0.5"),
            pytest.param({"overlap": 0.8}, id="overlap 0.8"),
            pytest.param({"stride": (3, 1, 4)}, id="uneven stride"),
            pytest.param({"stride": (6, 8, 5)}, id="no overlap"),
        ],
    )
    def test_without_resampling_the_identity_model_recovers_the_input(
        self, spacing: dict[str, Any]
    ) -> None:
        volume = random_volume((2, 13, 17, 19))
        pipeline = complete_pipeline(volume, patch_size=(6, 8, 5), **spacing)
        np.testing.assert_allclose(pipeline.finalize(), volume, rtol=1e-5)

    @pytest.mark.parametrize(
        ("options", "expected_working_shape"),
        [
            pytest.param(
                {"source_spacing": 1.0, "target_spacing": (2.0, None, None)},
                (51, 6, 5),
                id="101 voxels at 1.0 to 2.0",
            ),
            pytest.param({"target_shape": (13, 9, 2)}, (13, 9, 2), id="mixed with undersized"),
        ],
    )
    def test_finalize_returns_exactly_the_original_shape(
        self, options: dict[str, Any], expected_working_shape: tuple[int, int, int]
    ) -> None:
        pipeline = complete_pipeline(random_volume((101, 6, 5)), **options)
        assert pipeline.working_shape == expected_working_shape
        assert pipeline.finalize().shape == (1, 101, 6, 5)

    @pytest.mark.usefixtures("no_resampling")
    def test_single_input_channel_can_give_many_output_channels(self) -> None:
        volume = random_volume((6, 7, 8))
        pipeline = PatchPipeline(volume, patch_size=4)
        channel_scales = np.arange(7, dtype=np.float32)[:, None, None, None]
        for indices, patches in pipeline.batches(5):
            pipeline.feed(indices, patches * channel_scales)

        result = pipeline.finalize()
        assert pipeline.n_channels_out == 7
        assert result.shape == (7, 6, 7, 8)
        np.testing.assert_allclose(result, volume * channel_scales, rtol=1e-6)


class TestPreresampledVolume:
    @pytest.mark.parametrize("working_shape", [(5, 8, 11), (19, 30, 41)])
    def test_matches_resampling_inside_the_pipeline(
        self, working_shape: tuple[int, int, int]
    ) -> None:
        volume = random_volume((2, 11, 17, 23))
        resampled = resample_channels(
            volume,
            out_shape=working_shape,
            interpolator="linear",
            antialias=False,
            num_threads=None,
        )
        internal = complete_pipeline(volume, target_shape=working_shape)
        external = complete_pipeline(resampled, original_shape=(11, 17, 23))

        assert external.original_shape == (11, 17, 23)
        assert external.working_shape == working_shape
        np.testing.assert_array_equal(external.finalize(), internal.finalize())

    def test_targets_apply_to_the_given_volume(self) -> None:
        pipeline = complete_pipeline(
            np.full((10, 12, 14), 3.5, dtype=np.float32),
            original_shape=(20, 24, 7),
            source_spacing=(2.0, None, None),
            target_spacing=(4.0, None, None),
            target_shape=(None, 6, None),
        )
        assert pipeline.working_shape == (5, 6, 14)
        assert pipeline.effective_spacing == (4.0, None, None)
        np.testing.assert_allclose(pipeline.finalize(), 3.5, rtol=1e-6)


class TestUndersizedAxesOutput:
    @pytest.mark.usefixtures("no_resampling")
    def test_padding_is_cropped_from_the_output(self) -> None:
        volume = random_volume((2, 3, 20, 2))
        pipeline = complete_pipeline(volume, patch_size=(8, 8, 5), pad_mode="reflect")
        np.testing.assert_allclose(pipeline.finalize(), volume, rtol=1e-5)

    def test_pad_values_do_not_reach_the_inverse_resample(self) -> None:
        # with pad values far from the data, any padded voxel left in the inverse would show
        pipeline = complete_pipeline(
            np.ones((11, 17, 23), dtype=np.float32),
            patch_size=(8, 8, 8),
            target_shape=(5, 30, 3),
            pad_value=-100.0,
        )
        assert pipeline.padding == ((1, 2), (0, 0), (2, 3))
        np.testing.assert_allclose(pipeline.finalize(), 1.0, rtol=1e-6)


class TestFinalization:
    @pytest.fixture(params=[{}, {"target_shape": (9, 5, 6)}], ids=["fast path", "resampled"])
    def options(self, request: pytest.FixtureRequest) -> dict[str, Any]:
        options: dict[str, Any] = request.param
        return options

    @pytest.fixture
    def pipeline(self, options: dict[str, Any]) -> PatchPipeline:
        return complete_pipeline(random_volume((2, 6, 7, 8)), **options)

    def test_working_output_is_a_cropped_view_normalized_once(self) -> None:
        volume = random_volume((2, 3, 7, 8))
        pipeline = complete_pipeline(volume)

        first = pipeline.working_output()
        assert first.shape == (2, 3, 7, 8)
        assert np.shares_memory(first, accumulator(pipeline))
        np.testing.assert_allclose(first, volume, rtol=1e-6)
        np.testing.assert_array_equal(pipeline.working_output(), first)

    def test_finalize_after_working_output_gives_the_same_result(
        self, options: dict[str, Any]
    ) -> None:
        volume = random_volume((2, 6, 7, 8))
        direct = complete_pipeline(volume, **options).finalize()
        inspected = complete_pipeline(volume, **options)
        inspected.working_output()
        np.testing.assert_array_equal(inspected.finalize(), direct)

    @pytest.mark.usefixtures("no_resampling")
    def test_fast_path_returns_a_copy_of_the_working_output(self) -> None:
        pipeline = complete_pipeline(random_volume((2, 6, 7, 8)))
        working = pipeline.working_output()
        result = pipeline.finalize()
        assert result.dtype == np.float32
        assert not np.shares_memory(result, working)
        np.testing.assert_array_equal(result, working)

    def test_finalize_writes_into_a_memmap(self, options: dict[str, Any], tmp_path: Path) -> None:
        volume = random_volume((2, 6, 7, 8))
        expected = complete_pipeline(volume, **options).finalize()
        out = np.memmap(tmp_path / "out.dat", dtype=np.float32, mode="w+", shape=(2, 6, 7, 8))

        result = complete_pipeline(volume, **options).finalize(out=out)
        assert result is out
        np.testing.assert_array_equal(out, expected)

    @pytest.mark.parametrize(
        ("out", "error", "message"),
        [
            pytest.param(
                np.empty((2, 6, 7, 9), dtype=np.float32),
                ValueError,
                r"\(C_out, \*original_shape\) = \(2, 6, 7, 8\), got \(2, 6, 7, 9\)",
                id="shape",
            ),
            pytest.param(
                np.empty((6, 7, 8), dtype=np.float32), ValueError, "got \\(6, 7, 8\\)", id="ndim"
            ),
            pytest.param(
                np.empty((2, 6, 7, 8)), TypeError, "float32 numpy array, got float64", id="dtype"
            ),
            pytest.param([0.0], TypeError, "float32 numpy array, got list", id="not an array"),
        ],
    )
    def test_invalid_out_raises_and_finalize_can_be_retried(
        self, pipeline: PatchPipeline, out: Any, error: type, message: str
    ) -> None:
        with pytest.raises(error, match=message):
            pipeline.finalize(out=out)
        assert pipeline.finalize().shape == (2, 6, 7, 8)

    def test_num_threads_leaves_the_global_default_unchanged(self) -> None:
        default_threads = sitk.ProcessObject.GetGlobalDefaultNumberOfThreads()
        pipeline = complete_pipeline(
            random_volume((6, 7, 8)), target_shape=(9, 5, 6), num_threads=default_threads + 1
        )
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == default_threads
        pipeline.finalize()
        assert sitk.ProcessObject.GetGlobalDefaultNumberOfThreads() == default_threads


class TestLifecycle:
    @pytest.fixture
    def pipeline(self) -> PatchPipeline:
        # 18 patches of 4³
        return PatchPipeline(random_volume((2, 6, 7, 8)), patch_size=4)

    @pytest.mark.parametrize("finish", ["working_output", "finalize"])
    @pytest.mark.parametrize(
        ("fed", "message"),
        [
            pytest.param(
                slice(0), r"18 of 18 .*: \[0, 1, .*, 9, \.\.\.\] \(18 in total\)", id="none"
            ),
            pytest.param(
                slice(0, 15), r"3 of 18 patches have not been fed yet: \[15, 16, 17\]", id="most"
            ),
        ],
    )
    def test_finishing_before_every_patch_is_fed_lists_the_missing(
        self, pipeline: PatchPipeline, finish: str, fed: slice, message: str
    ) -> None:
        indices = np.arange(len(pipeline))[fed]
        if indices.size:
            pipeline.feed(indices, pipeline[indices])
        with pytest.raises(RuntimeError, match=message):
            getattr(pipeline, finish)()
        assert pipeline.fed.sum() == indices.size

    def test_working_input_is_released_on_the_last_feed(self, pipeline: PatchPipeline) -> None:
        pipeline.feed(slice(0, 17), pipeline[:17])
        pipeline[0]
        pipeline.feed(17, pipeline[17])
        assert pipeline._working_input is None
        with pytest.raises(RuntimeError, match="working input was released"):
            pipeline[0]

    def test_feeding_after_completion_raises(self, pipeline: PatchPipeline) -> None:
        feed_identity_model(pipeline)
        with pytest.raises(IndexError, match="already fed"):
            pipeline.feed(0, np.zeros((2, 4, 4, 4)))

    @pytest.mark.parametrize("call_again", ["working_output", "finalize"])
    def test_finalize_is_one_shot(self, pipeline: PatchPipeline, call_again: str) -> None:
        feed_identity_model(pipeline)
        pipeline.finalize()
        assert pipeline._accumulator is None
        with pytest.raises(RuntimeError, match="finalize was already called"):
            getattr(pipeline, call_again)()
