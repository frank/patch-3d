"""Opt-in benchmark on a realistic volume, run with `poe benchmark`.

Stage timings are printed, not asserted. Peak memory is asserted against the arrays each
stage is expected to allocate, so an unintended full-volume temporary fails the run.
"""

import math
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import pytest
import scipy.ndimage

from patch_pipeline import PatchPipeline, RoiPipeline

pytestmark = [
    pytest.mark.benchmark,
    pytest.mark.skipif(sys.platform != "linux", reason="peak memory is read from /proc"),
]

MIB = 2**20
ORIGINAL_SHAPE = (300, 512, 512)
WORKING_SHAPE = (256, 256, 256)
PATCH_SIZE = (128, 128, 128)
N_CHANNELS_OUT = 10
BATCH_SIZE = 4
SOURCE_SPACING = (1.0, 0.5, 0.5)
TARGET_SPACING = 1.0
MASK_SHAPE = (75, 128, 128)
# page granularity and interpreter allocations
MEMORY_TOLERANCE_MIB = 32


@dataclass
class Stage:
    seconds: float = 0.0
    peak_mib: float = 0.0  # peak resident memory above the start of a call


def float32_mib(*shape: int) -> float:
    return math.prod(shape) * 4 / MIB


def resident_mib(field: str) -> float:
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) * 1024 / MIB
    raise KeyError(field)


@contextmanager
def measured(stage: Stage) -> Iterator[None]:
    """Add a call's duration to the stage, and keep the largest peak memory over its calls."""
    with open("/proc/self/clear_refs", "w") as clear_refs:
        clear_refs.write("5")  # reset the peak resident memory to the current one
    start_mib = resident_mib("VmRSS")
    start = time.perf_counter()
    yield
    stage.seconds += time.perf_counter() - start
    stage.peak_mib = max(stage.peak_mib, resident_mib("VmHWM") - start_mib)


def feed_stand_in_model(pipeline: PatchPipeline | RoiPipeline, *, stages: dict[str, Stage]) -> None:
    """Fetch and feed every patch in batches, measuring the fetch and feed stages."""
    all_indices = np.arange(len(pipeline))
    for start in range(0, len(pipeline), BATCH_SIZE):
        indices = all_indices[start : start + BATCH_SIZE]
        with measured(stages["fetch"]):
            patches = pipeline[indices]
        # stand-in model: the input channel repeated, kept out of the measured stages
        outputs = np.repeat(patches, N_CHANNELS_OUT, axis=1)
        with measured(stages["feed"]):
            pipeline.feed(indices, outputs)
        del patches, outputs


def test_realistic_volume_stages(capsys: pytest.CaptureFixture[str]) -> None:
    volume = np.random.default_rng(seed=0).random(ORIGINAL_SHAPE, dtype=np.float32)
    stages = {name: Stage() for name in ("construct", "fetch", "feed", "normalize", "finalize")}

    with measured(stages["construct"]):
        pipeline = PatchPipeline(
            volume, target_shape=WORKING_SHAPE, patch_size=PATCH_SIZE, overlap=0.5
        )
    del volume
    feed_stand_in_model(pipeline, stages=stages)
    with measured(stages["normalize"]):
        working_output = pipeline.working_output()
    working_channel = working_output[0].copy()
    del working_output
    with measured(stages["finalize"]):
        result = pipeline.finalize()

    scipy_start = time.perf_counter()
    scipy_channel = scipy.ndimage.zoom(
        working_channel,
        zoom=np.divide(ORIGINAL_SHAPE, WORKING_SHAPE),
        order=1,
        grid_mode=True,
        mode="nearest",
    )
    scipy_seconds = time.perf_counter() - scipy_start

    with capsys.disabled():
        print(f"\n{pipeline!r}")
        for name, stage in stages.items():
            print(f"{name:>10}: {stage.seconds:6.2f} s, peak +{stage.peak_mib:6.0f} MiB")
        print(
            f"inverse resample of one channel: "
            f"SimpleITK {stages['finalize'].seconds / N_CHANNELS_OUT:.2f} s, "
            f"scipy {scipy_seconds:.2f} s"
        )

    np.testing.assert_allclose(result[0], scipy_channel, atol=1e-5)
    original_channel_mib = float32_mib(*ORIGINAL_SHAPE)
    working_channel_mib = float32_mib(*WORKING_SHAPE)
    # SimpleITK copies the input channel into an image, and resamples into a new one
    expected_peak_mib = {
        "construct": original_channel_mib + 2 * working_channel_mib,
        "fetch": float32_mib(BATCH_SIZE, *PATCH_SIZE),
        "feed": float32_mib(N_CHANNELS_OUT, *WORKING_SHAPE)
        + float32_mib(N_CHANNELS_OUT, *PATCH_SIZE),
        "normalize": 0.0,
        "finalize": float32_mib(N_CHANNELS_OUT, *ORIGINAL_SHAPE)
        + working_channel_mib
        + original_channel_mib,
    }
    for name, budget_mib in expected_peak_mib.items():
        assert stages[name].peak_mib <= budget_mib + MEMORY_TOLERANCE_MIB, name


def test_roi_pipeline_against_full_volume(capsys: pytest.CaptureFixture[str]) -> None:
    volume = np.random.default_rng(seed=0).random(ORIGINAL_SHAPE, dtype=np.float32)
    roi_mask = np.zeros(MASK_SHAPE, dtype=bool)
    roi_mask[30:50, 50:80, 50:80] = True  # fits in one patch at the target spacing
    stage_names = ("construct", "fetch", "feed", "finalize")
    full_stages = {name: Stage() for name in stage_names}
    roi_stages = {name: Stage() for name in stage_names}

    with measured(full_stages["construct"]):
        full = PatchPipeline(
            volume,
            patch_size=PATCH_SIZE,
            source_spacing=SOURCE_SPACING,
            target_spacing=TARGET_SPACING,
        )
    feed_stand_in_model(full, stages=full_stages)
    with measured(full_stages["finalize"]):
        full.finalize()
    full_description = repr(full)
    del full

    with measured(roi_stages["construct"]):
        roi = RoiPipeline(
            volume,
            roi_mask=roi_mask,
            patch_size=PATCH_SIZE,
            source_spacing=SOURCE_SPACING,
            target_spacing=TARGET_SPACING,
        )
    feed_stand_in_model(roi, stages=roi_stages)
    with measured(roi_stages["finalize"]):
        roi.finalize()

    with capsys.disabled():
        print(f"\n{full_description}\n{roi!r}")
        print(f"{'':>10}  {'PatchPipeline':>24}  {'RoiPipeline':>24}")
        for name in stage_names:
            full_stage, roi_stage = full_stages[name], roi_stages[name]
            print(
                f"{name:>10}: "
                f"{full_stage.seconds:6.2f} s, peak +{full_stage.peak_mib:6.0f} MiB  "
                f"{roi_stage.seconds:6.2f} s, peak +{roi_stage.peak_mib:6.0f} MiB"
            )

    assert roi.coverage == 1.0
    region_shape = tuple(axis.stop - axis.start for axis in roi.region)
    region_channel_mib = float32_mib(*region_shape)
    patch_channel_mib = float32_mib(*PATCH_SIZE)
    # the mask is converted to float64 for placement
    mask_mib = math.prod(MASK_SHAPE) * 8 / MIB
    expected_peak_mib = {
        "construct": mask_mib + region_channel_mib + 2 * patch_channel_mib,
        "fetch": patch_channel_mib,
        "feed": 2 * float32_mib(N_CHANNELS_OUT, *PATCH_SIZE),
        # a zero fill leaves pages unallocated outside the region's z slices, which
        # transparent huge pages may allocate whole
        "finalize": float32_mib(N_CHANNELS_OUT, region_shape[0], *ORIGINAL_SHAPE[1:])
        + patch_channel_mib
        + region_channel_mib,
    }
    for name, budget_mib in expected_peak_mib.items():
        assert roi_stages[name].peak_mib <= budget_mib + MEMORY_TOLERANCE_MIB, name
