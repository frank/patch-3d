# patch-pipeline-3d

Patch-based inference on 3D volumes, with resampling in both directions. `PatchPipeline`:

1. resamples a volume from its original voxel grid to a working domain, chosen per axis by
   target spacing or target shape,
2. splits the working volume into overlapping patches on a regular grid,
3. lets you run any transform, typically a deep-learning model, on each patch, producing a
   multi-channel output patch,
4. blends the output patches back into one volume with Hann-window weighting,
5. resamples the result back onto exactly the original voxel grid.

Resampling is done by multithreaded SimpleITK and blending by in-place numpy arithmetic.
Output patches accumulate into a single preallocated volume, so overlapping data is never
stored twice. The only runtime dependencies are `numpy` and `SimpleITK`.

When the structure of interest always lies in a contained area, `RoiPipeline` serves a single
high-resolution patch over a region found by a coarse pass, as described in
[ROI pipeline](#roi-pipeline).

Install this package with:

```sh
pip install git+https://github.com/frank/patch-pipeline-3d.git
# or, in a uv project
uv add git+https://github.com/frank/patch-pipeline-3d.git
```

Append `@<tag or commit>` to the URL to pin a version.

Requires Python 3.12 or newer, `numpy>=1.26` and `SimpleITK>=2.3.1`.

> [!WARNING]
> **All per-axis arguments follow the axis order of the numpy array.** This covers spacings,
> shapes, patch sizes and strides. For an array from `sitk.GetArrayFromImage`, the order is
> `(z, y, x)`, but `image.GetSpacing()` returns `(x, y, z)`. Reverse it before passing it in,
> as in the example below. Passing it unreversed raises no error: it swaps the z and x
> spacings and silently resamples those axes to the wrong length.

## Usage

```python
import numpy as np
import SimpleITK as sitk
import torch

from patch_pipeline import PatchPipeline

image = sitk.ReadImage("ct.nii.gz")
volume = sitk.GetArrayFromImage(image)  # (z, y, x)
spacing = image.GetSpacing()[::-1]  # (x, y, z) -> (z, y, x)

pipeline = PatchPipeline(
    volume,
    patch_size=128,
    source_spacing=spacing,
    target_spacing=(1.5, None, None),  # z: 1.5 mm slices
    target_shape=(None, 256, 256),  # y, x: a 256 × 256 in-plane matrix
    overlap=0.5,
)

with torch.no_grad():
    for indices, patches in pipeline.batches(4):  # (B,), (B, C_in, 128, 128, 128)
        logits = model(torch.from_numpy(patches).cuda())
        pipeline.feed(indices, logits.softmax(dim=1).cpu())  # (B, C_out, 128, 128, 128)

probabilities = pipeline.finalize()  # (C_out, *volume.shape), float32

labels = sitk.GetImageFromArray(probabilities.argmax(axis=0).astype(np.uint8))
labels.CopyInformation(image)  # the output is on the original voxel grid
sitk.WriteImage(labels, "labels.nii.gz")
```

The pipeline works in index space and never needs origin or direction. The output is on
exactly the original voxel grid, so the original image header applies to it unchanged.

A 3D volume `(Z, Y, X)` is treated as a single channel. Pass a channel-first
`(C_in, Z, Y, X)` array for several input channels. Patches always carry the channel axis.

## Choosing the working domain

Each axis is set independently, by giving a value for it in at most one of `target_spacing`
or `target_shape`:

| `target_spacing[k]` | `target_shape[k]` | Working length of axis k | Needs `source_spacing[k]` |
|---|---|---|---|
| value | `None` | `max(1, round_half_up(N0 · source_spacing / target_spacing))` | yes |
| `None` | value | `target_shape[k]` | no |
| `None` | `None` | unchanged, `N0` | no |
| value | value | error | |

A scalar applies to all three axes, and `None` for a whole argument means `None` on every
axis. `source_spacing` is needed only on axes with a target spacing.

Because the working length is rounded to whole voxels, the spacing actually used differs
slightly from the requested one. `pipeline.effective_spacing` reports it. If no axis changes
length, resampling is skipped in both directions.

**Spacing tolerance.** `spacing_rtol` leaves an axis at its source spacing when that is already
close to the target, `|source_spacing − target_spacing| ≤ spacing_rtol · target_spacing`:

```python
pipeline = PatchPipeline(
    volume,
    patch_size=128,
    source_spacing=spacing,
    target_spacing=(1.5, None, None),
    spacing_rtol=(0.05, None, None),  # z: keep slices between 1.425 and 1.575 mm as they are
    target_shape=(None, 256, 256),
)
```

Resampling by a factor close to 1 blurs unevenly: slices that land between two original
slices are blended from both, those that land on one stay sharp. The tolerance avoids that,
and keeps the original slices exactly. On such an axis, `effective_spacing` is the source
spacing. The tolerance only saves time if no axis is resampled at all.
`1e-9` is added to `spacing_rtol` to absorb float error, so a spacing exactly on the bound
counts as within it. Only axes with a target spacing may have a tolerance, so a scalar
`spacing_rtol` needs a target spacing on every axis.

> [!IMPORTANT]
> Use a tolerance only if the model's training preprocessing did the same. A model trained on
> volumes resampled to exactly 1.5 mm sees slightly different data otherwise.

**Grid alignment.** Resampling is *extent-preserving*: the original and working grids
span the same physical extent, the convention of `scipy.ndimage.zoom(grid_mode=True)`, torch's
`align_corners=False` and `skimage.transform.resize`. On an axis with `N0` original and `N1`
working voxels, working voxel `j` lies at original index `(j + 0.5) · N0 / N1 − 0.5`. Every
resampled voxel therefore falls within the data, and there are no zero-filled border slices.
`finalize` resamples back to the stored original shape, never to a shape recomputed from
spacings, so the output shape always matches the input exactly. For example, 101 voxels at
1 mm become 51 voxels at 2 mm, and come back as 101, not 102.

## Volumes resampled beforehand

If you resampled the volume yourself, pass it with the spatial shape of the original grid,
and `finalize` returns there:

```python
pipeline = PatchPipeline(resampled_volume, patch_size=128, original_shape=image.GetSize()[::-1])
```

`original_shape` defaults to the shape of `volume`. The targets, `source_spacing` and
`effective_spacing` always refer to the volume you pass in, so with targets given as well, it
is resampled again, and `finalize` maps the output straight back to `original_shape`.

> [!WARNING]
> **Your resampling must follow the extent-preserving convention above**, as
> `skimage.transform.resize`, torch's `align_corners=False` and
> `scipy.ndimage.zoom(grid_mode=True)` do. The pipeline cannot check this. Two common
> alternatives do not: `scipy.ndimage.zoom` with its default `grid_mode=False`, and the usual
> SimpleITK recipe that keeps the origin and sets the target spacing. With either, the output
> comes back shifted, for example by about 1.5 original voxels after a 4× downsampling.

## Patch grid

- `patch_size` is an int or a 3-tuple; patches need not be cubic.
- Give either `stride` or `overlap`, not both. Without either, `overlap=0.5`.
- `stride` is a *maximum*: patch starts are spaced evenly, with the first patch at 0 and the
  last flush with the end of the volume, so consecutive starts may be closer than `stride`.
  It may not exceed `patch_size`, which would leave voxels uncovered.
- `overlap` is a *minimum*, a fraction in `[0, 1)`, turned into the stride
  `floor(patch_size · (1 − overlap))`.
- An axis whose working length is shorter than the patch is padded symmetrically to the patch
  size, with `np.pad(mode=pad_mode)` and `pad_value` for the default `"constant"` mode. The
  padding is cropped before the output is resampled back.

Patches are addressed by a flat index `0 ≤ i < len(pipeline)`, in C order over
`pipeline.grid_shape`. `np.unravel_index(i, pipeline.grid_shape)` gives grid coordinates, and
`pipeline.locations[i]` the start voxel of patch `i` in padded working coordinates.
Multi-dimensional indexing like `pipeline[iz, iy, ix]` is deliberately not supported.

Where patches overlap, each output is weighted with a Hann window that is strictly positive
up to the patch edges, and the weighted sum is divided by the total weight at each voxel.
A voxel covered by a single patch keeps that patch's output exactly.

## Fetching and feeding

`pipeline[idx]` follows numpy indexing semantics on the flat index:

| `idx` | Returns |
|---|---|
| int, possibly negative | one patch, `(C_in, *patch_size)` |
| slice, int sequence or array, boolean mask of length `N` | a batch, `(B, C_in, *patch_size)` |

Fetched patches are float32 copies, so modifying them never affects the pipeline. Patches can
be fetched in any order and any number of times. `for i, patch in pipeline` and
`pipeline.batches(batch_size)` iterate over all of them in order.

`pipeline.feed(idx, outputs)` takes the same index forms, with outputs of shape
`(C_out, *patch_size)` for one patch or `(B, C_out, *patch_size)` for a batch. Outputs can be
anything `np.asarray` accepts, including CPU torch tensors, and are cast to float32. The first
feed sets `C_out`, which can differ from `C_in`. Every patch must be fed **exactly once**:
feeding a patch twice raises. Patches can be fed in any order. A rejected call raises before
anything is accumulated, so the pipeline is left unchanged.

`pipeline.fed`, `pipeline.is_complete` and `len(pipeline)` track progress.

## Finalizing

- `pipeline.finalize(out=None)` returns the blended output resampled to the original domain,
  `(C_out, *original_shape)`, float32. Pass a preallocated float32 `out`, for instance an
  `np.memmap`, to write very large outputs straight to disk. `finalize` can be called once.
- `pipeline.working_output()` returns the blended output in the working domain,
  `(C_out, *working_shape)`, as a view into the pipeline's buffer. It is useful for debugging,
  or if you want to post-process before resampling.

Both raise if any patch has not been fed, and list the missing indices.

To save memory, the pipeline releases each buffer when it is no longer needed. It drops the
working input when the last patch is fed, after which fetching raises. It drops the blending
buffer at the end of `finalize`, so keep the array that `finalize` returns.

## ROI pipeline

When the structure to segment always lies in a contained area, there is no need to tile the
whole volume with high-resolution patches. Two cheap passes are enough:

1. **Coarse pass.** A `PatchPipeline` resamples the whole volume to a low resolution, usually
   as a single patch, and a model segments the region of interest (ROI). `working_output()`
   returns the coarse result without resampling it back up.
2. **Fine pass.** A `RoiPipeline` takes the original volume and the low-resolution ROI mask,
   places one high-resolution patch over the ROI and serves only that patch. It returns the
   model's output on the original voxel grid, with every voxel outside the patch's region set
   to a fill value.

```python
import numpy as np
import SimpleITK as sitk

from patch_pipeline import PatchPipeline, RoiPipeline

image = sitk.ReadImage("ct.nii.gz")
volume = sitk.GetArrayFromImage(image)  # (z, y, x)
spacing = image.GetSpacing()[::-1]

coarse = PatchPipeline(volume, patch_size=96, source_spacing=spacing, target_spacing=4.0)
for indices, patches in coarse.batches(2):
    coarse.feed(indices, roi_model(patches))
roi_mask = coarse.working_output()[1] > 0.5  # low resolution, never resampled up
del coarse

fine = RoiPipeline(
    volume,
    roi_mask=roi_mask,
    patch_size=128,
    source_spacing=spacing,
    target_spacing=0.8,
)
for indices, patches in fine.batches(1):
    fine.feed(indices, model(patches))

probabilities = fine.finalize(fill_value=(1.0, 0.0, 0.0))  # (C_out, *volume.shape)
# or write only the region into your own array:
# fine.finalize_region(out=my_volume[:, *fine.region])
```

The fine pass never resamples or stores the whole volume at high resolution: it crops the
volume to the region and hands the crop to an inner `PatchPipeline`. Its time and memory scale
with the patch, apart from the full-size output of `finalize`.

`RoiPipeline` takes the same arguments as `PatchPipeline`, except `stride` and `overlap`
(there is a single patch), `original_shape` (the volume must be on the original grid) and
`target_shape` (the patch is set by `target_spacing` only). Interpolation, anti-aliasing,
padding, `num_threads` and concurrency work as for `PatchPipeline`, applied to the region.

### Placing the patch

**Region size.** The region is the block of original voxels that one patch spans:
`round_half_up(patch_size · target_spacing / source_spacing)` on axes with a target spacing,
and `patch_size` on axes without one or kept by `spacing_rtol`. It depends only on the patch
and the spacings, never on the size of the volume. For example, a 128³ patch at 0.8 mm over a
CT at `(2.5, 0.7, 0.7)` mm spans a 41×146×146 region. `pipeline.region` gives it as slices and
`pipeline.effective_spacing` the spacing actually used, `source_spacing · region_length /
working_length`, which differs from the target by rounding.

- An axis is resampled exactly when `PatchPipeline` would resample it, with one exception: on
  an axis that `PatchPipeline` would leave at its length only because of rounding over the
  whole volume, the patch is still resampled, so it is treated the same whatever the volume
  size. Near a factor of 1, use `spacing_rtol` to avoid resampling.
- On an axis shorter than the region, the region is the whole axis. It is resampled to the
  length `PatchPipeline` would give it, which never exceeds the patch, and padded to the patch
  with `pad_mode` and `pad_value`.

**Placement.** The patch goes where it covers the most mask weight. Among tied positions, it
goes where its centre is closest to the mask centroid, so a ROI smaller than the patch ends up
centred in it, as far as the volume borders allow. The placement is accurate to about half a
mask voxel, the resolution of the mask.

**Coverage.** `pipeline.coverage` is the fraction of the total mask weight inside the region,
measured at mask resolution. At 1.0 the whole ROI fits in the patch. Below that, part of the
ROI lies outside the region and gets `fill_value` instead of a prediction. Check `coverage`
and fall back to a full `PatchPipeline` at the fine spacing when it is low.

`roi_mask` is a 3D `(Z_m, Y_m, X_m)` array of bool, integer or floating weights, finite and
non-negative, with at least one positive voxel. A soft mask, such as the coarse
probabilities themselves, weights voxels by their value. Select one channel of a multi-channel
output, for example `working_output()[1]`. The pipeline keeps no reference to the mask after
construction.

> [!WARNING]
> **`roi_mask` must cover the same physical extent as `volume`**, at any shape. Mask position
> `e` maps to volume position `e · N0 / M`, with voxel `i` spanning `[i, i + 1)`. This holds
> for the `working_output()` of a `PatchPipeline` built on the same volume, and for any mask
> resampled with the extent-preserving convention described under
> [Volumes resampled beforehand](#volumes-resampled-beforehand). The pipeline cannot check it,
> and a mask on another convention places the patch off target.

### Fetching, feeding and finalizing

Indexing, iteration, `batches`, `feed`, `fed`, `is_complete`, `grid_shape` and the channel
counts behave as in `PatchPipeline`. Today there is always one patch: `len(pipeline) == 1` and
`grid_shape == (1, 1, 1)`. `locations`, `working_shape` and `working_output()` refer to the
region's working domain, not the whole volume's.

- `pipeline.finalize(out=None, fill_value=0.0)` returns `(C_out, *original_shape)`, float32,
  with the region filled from the model's output and every other voxel set to `fill_value`.
  `fill_value` is a scalar for every channel or one value per output channel, and must be
  finite. `out` may be a preallocated float32 array or an `np.memmap`.
- `pipeline.finalize_region(out=None)` returns only the region, `(C_out, *region_shape)`. Pass
  a view to write it straight into your own array, which leaves the rest of that array
  untouched:

  ```python
  output_volume = np.zeros((n_classes, *volume.shape), dtype=np.float32)
  fine.finalize_region(out=output_volume[:, *fine.region])
  ```

Either method can be called once, and calling one uses up the other. A rejected call, for
example with a wrong-length `fill_value` or with patches still missing, raises before
anything is written, so `out` and the pipeline are left unchanged.

> [!IMPORTANT]
> **Choose `fill_value` to mark the background explicitly.** With softmax outputs and the
> default fill of 0, every channel is 0 outside the region, so `argmax` gives channel 0 there,
> which is only correct if channel 0 is the background. Pass, for example,
> `fill_value=(1.0, 0.0, 0.0)`. For logits, use a large negative number rather than `-inf`,
> which a softmax turns into NaN.

### Memory and time

All working data is float32, and the patch buffers do not depend on the size of the volume:

| Array | Size | Lifetime |
|---|---|---|
| working input | `C_in · prod(patch_size) · 4` bytes | construction until the last feed |
| SimpleITK temporaries | about one channel of the region plus one of the patch | during each resampling |
| blending buffer | `C_out · prod(patch_size) · 4` bytes | first feed until the end of finalizing |
| output of `finalize` | `C_out · prod(original_shape) · 4` bytes | returned to you |
| output of `finalize_region` | `C_out · prod(region_shape) · 4` bytes | returned, or written into your array |

A 128³ patch takes about 8.4 MB per channel. The full-size output of `finalize` usually
dominates. With `out=None` and a fill of 0 in every channel, it is allocated with `np.zeros`,
and the operating system typically leaves the pages outside the region unallocated until
something writes to them. Any nonzero fill writes the whole output.

Resampling into the patch takes time in proportion to the region and the patch, and back in
proportion to `C_out` times the region. Placement is linear in the mask voxels, and a nonzero
fill linear in the output. `"bspline"` becomes affordable, because its single-threaded
prefilter only runs over the region.

### Differences from a whole-volume pass

The fine pass crops first and resamples the crop, so its patch is not exactly a block cut out
of the whole volume resampled to `target_spacing`:

- **The patch grid is not the whole-volume working grid.** It can be offset from it by a
  fraction of a voxel, and its spacing is `effective_spacing` rather than exactly
  `target_spacing`. Over the whole patch, the difference is at most half an original voxel.
- **Interpolation at the region border.** When upsampling, working voxels at the region border
  extrapolate within half a voxel from the nearest neighbour, instead of using the data just
  outside the crop. With `antialias_in`, smoothing near the crop edge sees the edge instead of
  the neighbouring data.
- **Hard edge outside the region.** The output switches abruptly to `fill_value` at the region
  border, and anything the model would have predicted outside it is lost.

## Interpolation and anti-aliasing

`interp_in` resamples the input and `interp_out` the output. They are independent and both
default to `"linear"`.

| Name | SimpleITK interpolator | Notes |
|---|---|---|
| `"nearest"` | `sitkNearestNeighbor` | for hard labels |
| `"linear"` | `sitkLinear` | the default; suits intensities and soft outputs |
| `"bspline"` | `sitkBSpline` (cubic) | overshoots; slow, see below |
| `"lanczos"` | `sitkLanczosWindowedSinc` | overshoots; does not preserve constants |

- **Overshoot.** `"bspline"` and `"lanczos"` can produce values outside the input range, such
  as negative intensities or probabilities above 1. The pipeline never clamps them; clip the
  output yourself if that matters.
- **`"lanczos"` does not preserve constants.** The windowed sinc does not sum to one, so a
  constant volume comes back about 0.4% off, in the interior as well as at the borders.
  `"nearest"`, `"linear"` and `"bspline"` reproduce constants exactly.
- **`"bspline"` is slow and memory-hungry.** ITK computes the B-spline coefficients with a
  single-threaded prefilter, and stores them in float64, twice the size of a float32 channel.
  On a 6-core machine, resampling a 300×512×512 volume to 256³ takes about 13 s with
  `"bspline"` against 0.13 s with `"linear"`, and resampling back with `"bspline"` takes
  about 6 s per output channel.
- The package does not post-process outputs. Whether to resample probabilities and take the
  argmax afterwards, as in the example, or to take the argmax in the working domain and
  resample labels with `interp_out="nearest"`, is up to you.

`antialias_in` and `antialias_out` (default `False`) Gaussian-smooth the data along each
downsampled axis before resampling it, with `sigma = (N_in / N_out − 1) / 2` input voxels,
the convention of `skimage.transform.resize`. Axes shorter than 4 voxels are left unsmoothed.

> [!IMPORTANT]
> Enable anti-aliasing only if the model's training preprocessing did the same. Preprocessing
> that differs from training silently degrades results.

## Performance and memory

All working data is float32. The pipeline holds at most:

| Array | Size | Lifetime |
|---|---|---|
| working input | `C_in · prod(padded_shape) · 4` bytes | construction until the last feed |
| blending buffer | `C_out · prod(padded_shape) · 4` bytes | first feed until the end of `finalize` |
| output | `C_out · prod(original_shape) · 4` bytes | returned to you |

Peak memory is reached in `finalize`, with the blending buffer, the output and one channel of
SimpleITK temporaries alive at once. For example, from a 300×512×512 original with a
300×256×256 working shape and 10 output channels, the blending buffer is about 0.79 GB and
the output about 3.15 GB. That is why `finalize(out=...)` accepts a memmap.

Feeding takes time in proportion to the voxels fed, not the number of patches. Resampling
back is usually the costliest stage, because it writes `C_out` channels onto the larger
original grid. It runs channel by channel, so its temporaries stay at one channel.

`num_threads` limits the threads SimpleITK uses for resampling, which helps on shared
machines. SimpleITK only honours its process-global thread default, so the pipeline sets
that default while it resamples, in the constructor and in `finalize`, and restores it
afterwards, also when an error is raised. SimpleITK code you run between those calls uses your
own setting. With `num_threads=None`, the default, the global setting is never touched.

## Concurrency

- A `PatchPipeline` is **not thread-safe**. Use each instance from one thread.
- With `num_threads` set, another thread that runs SimpleITK while the pipeline resamples
  sees the pipeline's thread count. Other processes are never affected.
- **Do not hand the pipeline to `DataLoader` workers** or other processes. Pickling it
  copies the working volume into each worker. Fetching is cheap slicing, so do it in the main
  process and keep workers for loading and preprocessing whole volumes.

## Scope

The package deliberately does not:

- post-process outputs, for example with argmax or thresholds. `RoiPipeline`'s `fill_value`
  only defines the voxels that were never processed, and leaves the model's output in the
  region untouched;
- handle world-space metadata (origin, direction), which you reattach from the original image;
- support outputs with a different spatial size than the input patch, or tuple, dict and
  deep-supervision outputs;
- accumulate on the GPU, sample random patches for training, or offer windows other than
  Hann.

## Development

Requires [uv](https://docs.astral.sh/uv/). Works on Linux, macOS and Windows.

```sh
uv sync              # create the environment
uv run poe gate      # format check, lint, typecheck, tests
```

| Task | Purpose |
|---|---|
| `uv run poe format` | rewrite files to satisfy the formatter and fixable lint rules |
| `uv run poe lint` | check formatting and lint rules without modifying files |
| `uv run poe typecheck` | mypy over `src` and `tests` |
| `uv run poe test` | test suite, excluding benchmarks |
| `uv run poe benchmark` | only the opt-in benchmark tests (Linux only) |
| `uv run poe gate` | everything that must pass before a commit |

CI runs the tests at both ends of the supported range: with the locked, newest dependency
versions on Python 3.12 and 3.14, and at the declared floors on Python 3.12
(`numpy` 1.26.0, `SimpleITK` 2.3.1, `scipy` 1.11.2, `pytest` 8.0.0).
