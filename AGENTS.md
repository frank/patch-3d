# Repository Overview

`patch-pipeline-3d` (import name `patch_pipeline`) runs patch-based inference on 3D volumes. It has two public classes. `PatchPipeline` resamples a volume to a working domain, serves overlapping patches, accumulates the fed-back outputs with Hann-window blending, and resamples the result onto exactly the original voxel grid. `RoiPipeline` places one high-resolution patch over a region of interest given by a low-resolution mask, crops the volume to that region and delegates to an inner `PatchPipeline`, then writes the output back onto the original grid with a fill value outside the region. The README is the user-facing spec, so read it before changing behaviour and keep it in sync.

- `src/patch_pipeline/_pipeline.py`: `PatchPipeline`, which handles indexing, feeding, finalizing and buffer lifetime.
- `src/patch_pipeline/_roi_pipeline.py`: `RoiPipeline`, which composes a `PatchPipeline` (no subclassing), forwards its indexing and feeding API, and adds `finalize` with a fill value and `finalize_region`.
- `src/patch_pipeline/_roi.py`: pure functions for the ROI pipeline: mask validation, region and working shapes, patch placement by sliding window sums, and fill values.
- `src/patch_pipeline/_geometry.py`: argument validation and per-axis broadcasting, working shape, stride, padding and patch grid.
- `src/patch_pipeline/_resampling.py`: SimpleITK resampling and anti-aliasing. All per-axis values are in numpy `(z, y, x)` order, and only this module flips them to SimpleITK's `(x, y, z)`.
- `src/patch_pipeline/_blending.py`: Hann windows and per-axis weight normalization.
- `tests/`: one test module per source module, including `test_roi.py` and `test_roi_pipeline.py`. The resampling tests check against scipy as an independent reference, and the ROI pipeline tests check against a `PatchPipeline` built by hand on the same crop. `test_benchmark.py` is opt-in.

Runtime dependencies are only `numpy` and `SimpleITK`, and supported versions are Python >= 3.12. The environment is managed with `uv`, and tasks run through `poe`:

- `uv run poe format`: ruff format and fixable lint fixes.
- `uv run poe gate`: lint, strict mypy over `src` and `tests`, and tests. It must pass before work is done.
- `uv run poe benchmark`: the slow benchmark tests.

# Code Style

This package needs to be high quality in the following ways:
  - It should be, above all, readable. Always prefer expressive variable names and favour readable expressions.
  - Always use type annotations in function arguments and return types. Use type annotations in variable declarations sparingly, and only when it is helpful to read the code.
  - Always prefer passing keyword arguments to funcitons instead of positional.
  - Don't write overly elaborate or too long comments. When the code is easily readable, it's self-documenting. Inline comments should start with lowercase letters, multi-line comments can follow proper grammar/capitalization.
  - Don't write unnecessary code, aim to write the minimum amount of code for a purpose. Less code, readable, and kept modular, is easier to maintain.

# Development Process

- Architecture and design in this package are important. Whenever working on features that carry choices that affect usage or design downstream, make those choices explicit, so I am aware of them and have a chance to steer them.
- Never commit or stage anything. I will stage and commit myself as a way of reviewing and understanding the code.