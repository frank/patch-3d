"""The PatchPipeline class, assembling geometry, resampling and blending into one object."""

from collections.abc import Iterator, Sequence
from typing import Any, Literal, SupportsIndex, cast, get_args

import numpy as np
import numpy.typing as npt

from patch_pipeline._blending import grid_normalizers, normalize_in_place, patch_window
from patch_pipeline._geometry import (
    Padding,
    PerAxis,
    Shape,
    broadcast_per_axis,
    effective_spacing,
    grid_starts,
    non_negative_float,
    optional,
    overlap_fraction,
    patch_locations,
    positive_float,
    positive_int,
    resolve_stride,
    resolve_working_shape,
    undersize_padding,
)
from patch_pipeline._resampling import interpolator_name, resample_channels

type AxisArgument[T] = T | Sequence[T]
# numpy indexing semantics on the flat patch index space
type PatchIndex = (
    SupportsIndex | slice | Sequence[int] | Sequence[bool] | npt.NDArray[np.integer[Any] | np.bool_]
)

# np.pad's "empty" mode is left out, since it would show uninitialized memory to the model
PadMode = Literal[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
]
PAD_MODES: tuple[str, ...] = get_args(PadMode)


class PatchPipeline:
    """Resample a volume to a working domain, serve it as overlapping patches, and stitch the
    patch outputs back into the original domain.

    Construction resamples the volume and keeps no reference to the caller's array. Every patch
    is then fetched by flat index (`pipeline[i]`, `iter(pipeline)` or `batches`), transformed
    by the caller, and fed back exactly once with `feed`. Overlapping outputs are blended with
    a Hann window, and `finalize` resamples the result onto the original voxel grid.

    Conventions:

    - Axis order: every per-axis argument follows the axis order of the numpy array. For an
      array from `sitk.GetArrayFromImage` that is (z, y, x), while `image.GetSpacing()`
      returns (x, y, z) and must be reversed.
    - Grid alignment: resampling is extent-preserving, like `scipy.ndimage.zoom` with
      `grid_mode=True` or torch's `align_corners=False`. Both grids span the same extent, so
      working voxel j lies at original index (j + 0.5) * N0 / N1 - 0.5. The output of
      `finalize` lies on exactly the original voxel grid, so the original image header
      (origin, spacing, direction) applies to it unchanged. A volume resampled beforehand and
      passed with `original_shape` must follow the same convention, which cannot be checked.
    - Scope: outputs are not post-processed. Whether to resample probabilities or labels is
      the caller's choice, through `interp_out`. Outputs are single arrays with the spatial
      size of the input patch.
    - Concurrency: an instance is not thread-safe. Do not hand it to `DataLoader` workers or
      other processes, since pickling copies the working volume into each of them; fetching
      is cheap slicing, so do it in the main process.

    Args:
        volume: (Z, Y, X), or channel-first (C_in, Z, Y, X), of integer or floating dtype.
            All working data is float32.
        patch_size: Patch size per axis, or one size for all axes.
        original_shape: Spatial shape that `finalize` resamples to, by default the shape of
            `volume`. Give it when `volume` was already resampled from an original grid.
        source_spacing: Voxel spacing of `volume`, needed only on axes with a target spacing.
        target_spacing: Working spacing per axis. With N_in voxels in `volume`, the working
            length is N_in * source_spacing / target_spacing rounded half up, and at least 1, so
            the spacing actually used, `effective_spacing`, differs slightly from the request.
        spacing_rtol: Relative tolerance per axis on the target spacing. An axis is left
            unresampled if |source_spacing - target_spacing| <= spacing_rtol * target_spacing,
            with 1e-9 added to spacing_rtol to absorb float error. Only axes with a target
            spacing may have one.
        target_shape: Working length per axis. An axis may have a target spacing or a target
            shape but not both, and an axis with neither is not resampled.
        stride: Maximum distance between patch starts, at most `patch_size`. Starts are spaced
            evenly from 0 to flush with the end, so the actual distance may be smaller.
        overlap: Minimum overlap fraction in [0, 1), turned into the stride
            floor(patch_size * (1 - overlap)). Give `stride` or `overlap`, not both; without
            either, `overlap` is 0.5.
        interp_in: Interpolator for resampling the input: "nearest", "linear", "bspline"
            (cubic) or "lanczos". "bspline" and "lanczos" can overshoot, e.g. below zero or
            above one, and values are never clamped. "lanczos" also changes a constant
            volume by about 0.4%. "bspline" is much slower than "linear" and needs more
            memory, as ITK computes its coefficients on one thread and stores them in float64.
        interp_out: Interpolator for resampling the output back, with the same choices.
        antialias_in: Gaussian-smooth the input along downsampled axes before resampling, with
            sigma = (N_in / N_out - 1) / 2 input voxels. Axes shorter than 4 voxels are left
            unsmoothed. Enable only if the model's training preprocessing did the same, since
            mismatched preprocessing silently degrades results.
        antialias_out: The same for resampling the output back.
        pad_mode: `np.pad` mode for axes whose working length is shorter than the patch,
            which are padded symmetrically to the patch size. The padding is cropped again
            before the output is resampled back.
        pad_value: Fill value for `pad_mode="constant"`.
        num_threads: SimpleITK threads for resampling. SimpleITK honours only its
            process-global default, so it is set for the duration of the resampling in the
            constructor and in `finalize`, and restored afterwards, also on errors. Another
            thread running SimpleITK at the same moment sees the temporary value. None leaves
            the global default untouched.

    Raises:
        TypeError, ValueError: An argument is invalid.
    """

    def __init__(
        self,
        volume: npt.ArrayLike,
        *,
        patch_size: AxisArgument[int],
        original_shape: AxisArgument[int] | None = None,
        source_spacing: AxisArgument[float | None] = None,
        target_spacing: AxisArgument[float | None] = None,
        spacing_rtol: AxisArgument[float | None] = None,
        target_shape: AxisArgument[int | None] = None,
        stride: AxisArgument[int] | None = None,
        overlap: AxisArgument[float] | None = None,
        interp_in: str = "linear",
        interp_out: str = "linear",
        antialias_in: bool = False,
        antialias_out: bool = False,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        num_threads: int | None = None,
    ) -> None:
        volume = channel_first(numeric_array(volume, name="volume"))
        self._n_channels_in = len(volume)
        input_shape = spatial_shape(volume)
        self._original_shape = (
            input_shape
            if original_shape is None
            else broadcast_per_axis(original_shape, name="original_shape", convert=positive_int)
        )
        self._patch_size = broadcast_per_axis(patch_size, name="patch_size", convert=positive_int)
        known_spacing = broadcast_per_axis(
            source_spacing, name="source_spacing", convert=optional(positive_float)
        )
        self._working_shape = resolve_working_shape(
            input_shape=input_shape,
            source_spacing=known_spacing,
            target_spacing=broadcast_per_axis(
                target_spacing, name="target_spacing", convert=optional(positive_float)
            ),
            spacing_rtol=broadcast_per_axis(
                spacing_rtol, name="spacing_rtol", convert=optional(non_negative_float)
            ),
            target_shape=broadcast_per_axis(
                target_shape, name="target_shape", convert=optional(positive_int)
            ),
        )
        self._effective_spacing = effective_spacing(
            input_shape=input_shape,
            working_shape=self._working_shape,
            source_spacing=known_spacing,
        )
        max_stride = resolve_stride(
            patch_size=self._patch_size,
            stride=None
            if stride is None
            else broadcast_per_axis(stride, name="stride", convert=positive_int),
            overlap=None
            if overlap is None
            else broadcast_per_axis(overlap, name="overlap", convert=overlap_fraction),
        )
        self._padding = undersize_padding(
            working_shape=self._working_shape, patch_size=self._patch_size
        )
        interp_in = interpolator_name(interp_in, name="interp_in")
        self._interp_out = interpolator_name(interp_out, name="interp_out")
        self._antialias_out = antialias_out
        self._num_threads = optional(positive_int)(num_threads, name="num_threads")
        checked_pad_mode = pad_mode_name(pad_mode, name="pad_mode")

        if self._working_shape == input_shape:
            # astype always copies, so no reference to the caller's array is kept
            working_volume = volume.astype(np.float32)
        else:
            working_volume = resample_channels(
                volume,
                out_shape=self._working_shape,
                interpolator=interp_in,
                antialias=antialias_in,
                num_threads=self._num_threads,
            )
        self._working_input: npt.NDArray[np.float32] | None = pad_spatial_axes(
            working_volume, padding=self._padding, mode=checked_pad_mode, value=pad_value
        )

        self._padded_shape = spatial_shape(self._working_input)
        starts = grid_starts(
            padded_shape=self._padded_shape, patch_size=self._patch_size, stride=max_stride
        )
        starts_z, starts_y, starts_x = starts
        self._grid_shape = (len(starts_z), len(starts_y), len(starts_x))
        self._locations = patch_locations(starts)
        self._locations.flags.writeable = False
        self._window = patch_window(self._patch_size)
        self._normalizers = grid_normalizers(
            padded_shape=self._padded_shape, patch_size=self._patch_size, starts=starts
        )
        self._fed = np.zeros(len(self._locations), dtype=bool)
        self._n_channels_out: int | None = None
        # allocated on the first feed, once the number of output channels is known
        self._accumulator: npt.NDArray[np.float32] | None = None
        self._scratch: npt.NDArray[np.float32] | None = None
        self._normalized = False

    @property
    def original_shape(self) -> Shape:
        """The spatial shape that `finalize` returns to, by default that of the input volume."""
        return self._original_shape

    @property
    def working_shape(self) -> Shape:
        """The resampled shape, before padding."""
        return self._working_shape

    @property
    def padded_shape(self) -> Shape:
        """The working shape after padding the axes shorter than the patch."""
        return self._padded_shape

    @property
    def padding(self) -> Padding:
        """(before, after) padding per axis, cropped again from the output."""
        return self._padding

    @property
    def effective_spacing(self) -> PerAxis[float | None]:
        """The working spacing source_spacing * N_in / N1, or None where the source spacing is
        unknown."""
        return self._effective_spacing

    @property
    def patch_size(self) -> Shape:
        return self._patch_size

    @property
    def grid_shape(self) -> Shape:
        """Number of patches per axis. Flat patch indices run over it in C order, so
        `np.unravel_index(i, grid_shape)` gives the grid coordinates of patch i."""
        return self._grid_shape

    @property
    def locations(self) -> npt.NDArray[np.intp]:
        """Start voxel of each patch in padded working coordinates, shape (N, 3), read-only."""
        return self._locations

    @property
    def fed(self) -> npt.NDArray[np.bool_]:
        """A copy of which patches have been fed, shape (N,)."""
        return self._fed.copy()

    @property
    def is_complete(self) -> bool:
        return bool(self._fed.all())

    @property
    def n_channels_in(self) -> int:
        return self._n_channels_in

    @property
    def n_channels_out(self) -> int | None:
        """Output channels, taken from the first feed; None before it."""
        return self._n_channels_out

    def __len__(self) -> int:
        return len(self._locations)

    def __getitem__(self, idx: PatchIndex) -> npt.NDArray[np.float32]:
        """A float32 copy of one patch, (C_in, *patch_size), or of a batch,
        (B, C_in, *patch_size).

        `idx` follows numpy semantics on the flat patch indices: an integer, possibly
        negative, gives one patch, while a slice, an integer sequence or array, or a boolean
        mask of length N gives a batch. Patches can be fetched in any order and any number
        of times, until the last patch is fed and the working input is released.

        Raises:
            IndexError: The index is out of range or not one-dimensional.
            RuntimeError: The working input was already released.
        """
        if self._working_input is None:
            raise RuntimeError(
                "the working input was released when the last patch was fed, "
                "so patches can no longer be fetched"
            )
        flat = self._flat_indices(idx)
        patches = np.empty((flat.size, self._n_channels_in, *self._patch_size), dtype=np.float32)
        for patch, index in zip(patches, np.atleast_1d(flat), strict=True):
            patch[...] = self._working_input[:, *self._patch_region(index)]
        return patches[0] if flat.ndim == 0 else patches

    def feed(self, idx: PatchIndex, outputs: npt.ArrayLike) -> None:
        """Accumulate the window-weighted outputs of one patch, (C_out, *patch_size), or of a
        batch, (B, C_out, *patch_size).

        `idx` takes the same forms as for fetching, but must select at least one patch, and
        each patch is fed exactly once. `outputs` can be anything `np.asarray` accepts,
        such as a CPU torch tensor, and is cast to float32. The first feed sets C_out.
        The whole call is validated before anything is accumulated, so a rejected call
        leaves the pipeline unchanged.

        Raises:
            IndexError: The selection is invalid, empty, repeats a patch, or contains a
                patch that was already fed.
            TypeError, ValueError: `outputs` has a non-numeric dtype or the wrong shape.
        """
        flat = self._flat_indices(idx)
        indices = np.atleast_1d(flat)
        if indices.size == 0:
            raise IndexError("feed needs at least one patch index, got an empty selection")
        unique_indices, counts = np.unique(indices, return_counts=True)
        if np.any(counts > 1):
            raise IndexError(
                "patch indices must be unique within a feed, got repeats of "
                f"{index_summary(unique_indices[counts > 1])}"
            )
        already_fed = indices[self._fed[indices]]
        if already_fed.size > 0:
            raise IndexError(
                "each patch is fed exactly once, but these patches were already fed: "
                f"{index_summary(already_fed)}"
            )
        outputs = numeric_array(outputs, name="outputs")
        channel_axis = flat.ndim
        n_channels_out = self._n_channels_out
        if n_channels_out is None and outputs.ndim == channel_axis + 4:
            # the first feed sets the number of output channels
            n_channels_out = outputs.shape[channel_axis]
        expected_shape = (*flat.shape, n_channels_out, *self._patch_size)
        if outputs.shape != expected_shape:
            layout = "(B, C_out, *patch_size)" if flat.ndim else "(C_out, *patch_size)"
            raise ValueError(
                f"outputs must have shape {layout} = {expected_shape}, got {outputs.shape}"
            )

        if self._accumulator is None:
            self._n_channels_out = outputs.shape[channel_axis]
            self._accumulator = np.zeros(
                (self._n_channels_out, *self._padded_shape), dtype=np.float32
            )
            self._scratch = np.empty((self._n_channels_out, *self._patch_size), dtype=np.float32)
        batch_outputs = outputs if flat.ndim else outputs[None]
        # patches in one batch may overlap, so they are accumulated one at a time
        for index, patch_outputs in zip(indices, batch_outputs, strict=True):
            weighted = np.multiply(patch_outputs, self._window, out=self._scratch)
            self._accumulator[:, *self._patch_region(index)] += weighted
        self._fed[indices] = True
        if self.is_complete:
            # every patch has been fed, so none is fetched again
            self._working_input = None
            self._scratch = None

    def __iter__(self) -> Iterator[tuple[int, npt.NDArray[np.float32]]]:
        """Each flat patch index with its patch, in order."""
        for index in range(len(self)):
            yield index, self[index]

    def batches(
        self, batch_size: int
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]]:
        """Consecutive batches of patch indices and patches; the last batch may be smaller."""
        batch_size = positive_int(batch_size, name="batch_size")
        all_indices = np.arange(len(self), dtype=np.intp)
        for start in range(0, len(self), batch_size):
            indices = all_indices[start : start + batch_size]
            yield indices, self[indices]

    def working_output(self) -> npt.NDArray[np.float32]:
        """The normalized output in the working domain, (C_out, *working_shape), as a view into
        the accumulator. Normalization happens on the first call only.

        Raises:
            RuntimeError: Some patches have not been fed, or `finalize` was already called.
        """
        if not self.is_complete:
            missing = np.flatnonzero(~self._fed)
            raise RuntimeError(
                f"{missing.size} of {len(self)} patches have not been fed yet: "
                f"{index_summary(missing)}"
            )
        if self._accumulator is None:
            raise RuntimeError(
                "finalize was already called and released the accumulator; "
                "keep the array it returned"
            )
        if not self._normalized:
            normalize_in_place(self._accumulator, normalizers=self._normalizers)
            self._normalized = True
        unpadded_region = tuple(
            slice(before, before + length)
            for (before, _), length in zip(self._padding, self._working_shape, strict=True)
        )
        return self._accumulator[:, *unpadded_region]

    def finalize(self, out: npt.NDArray[np.float32] | None = None) -> npt.NDArray[np.float32]:
        """The output resampled to the original domain, (C_out, *original_shape), float32.

        It is written into `out` if given, which may be an np.memmap for very large outputs.
        Finalizing releases the accumulator, so it can be called only once.

        Raises:
            RuntimeError: Some patches have not been fed, or `finalize` was already called.
            TypeError, ValueError: `out` is not a float32 array of the output shape.
        """
        working = self.working_output()
        out = output_array(
            out, shape=(len(working), *self._original_shape), layout="(C_out, *original_shape)"
        )
        if self._working_shape == self._original_shape:
            out[...] = working
        else:
            resample_channels(
                working,
                out_shape=self._original_shape,
                interpolator=self._interp_out,
                antialias=self._antialias_out,
                num_threads=self._num_threads,
                out=out,
            )
        self._accumulator = None
        return out

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"original_shape={self.original_shape}, working_shape={self.working_shape}, "
            f"padded_shape={self.padded_shape}, patch_size={self.patch_size}, "
            f"grid_shape={self.grid_shape}, n_channels_in={self.n_channels_in}, "
            f"n_channels_out={self.n_channels_out}, fed={np.count_nonzero(self._fed)}/{len(self)})"
        )

    def _flat_indices(self, idx: PatchIndex) -> npt.NDArray[np.intp]:
        """Normalize an index with numpy semantics into a 0D or 1D array of flat patch indices."""
        try:
            flat = np.asarray(np.arange(len(self), dtype=np.intp)[idx])
        except IndexError as error:
            raise IndexError(
                f"invalid patch index for a pipeline with {len(self)} patches: {error}"
            ) from None
        if flat.ndim > 1:
            raise IndexError(
                "patch indices must be a scalar or one-dimensional, since patches are addressed "
                f"by flat index; got an index selecting shape {flat.shape}"
            )
        return flat

    def _patch_region(self, index: int) -> tuple[slice, ...]:
        """Spatial slices of a patch in padded working coordinates."""
        return tuple(
            slice(start, start + size)
            for start, size in zip(self._locations[index], self._patch_size, strict=True)
        )


def numeric_array(value: npt.ArrayLike, *, name: str) -> npt.NDArray[Any]:
    array = np.asarray(value)
    if not (np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.floating)):
        raise TypeError(f"{name} must have an integer or floating dtype, got {array.dtype}")
    return array


def channel_first(volume: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Validate a (Z, Y, X) or (C, Z, Y, X) volume and give it an explicit channel axis."""
    if volume.ndim not in (3, 4):
        raise ValueError(
            f"volume must be 3D (Z, Y, X) or 4D (C, Z, Y, X), got shape {volume.shape}"
        )
    if volume.size == 0:
        raise ValueError(f"volume must not be empty, got shape {volume.shape}")
    return volume[None] if volume.ndim == 3 else volume


def spatial_shape(volume: npt.NDArray[Any]) -> Shape:
    _, depth, height, width = volume.shape
    return depth, height, width


def index_summary(indices: npt.NDArray[np.intp], *, limit: int = 10) -> str:
    """The first few indices, for error messages that may concern thousands of patches."""
    shown = ", ".join(str(index) for index in indices[:limit])
    return f"[{shown}, ...] ({len(indices)} in total)" if len(indices) > limit else f"[{shown}]"


def output_array(
    out: npt.NDArray[np.float32] | None, *, shape: tuple[int, ...], layout: str
) -> npt.NDArray[np.float32]:
    """Validate a caller-supplied output array, or allocate one."""
    if out is None:
        return np.empty(shape, dtype=np.float32)
    if not isinstance(out, np.ndarray) or out.dtype != np.float32:
        description = out.dtype if isinstance(out, np.ndarray) else type(out).__name__
        raise TypeError(f"out must be a float32 numpy array, got {description}")
    if out.shape != shape:
        raise ValueError(f"out must have shape {layout} = {shape}, got {out.shape}")
    return out


def pad_mode_name(value: object, *, name: str) -> PadMode:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {value!r}")
    if value not in PAD_MODES:
        raise ValueError(f"{name} must be one of {', '.join(PAD_MODES)}, got {value!r}")
    return cast(PadMode, value)


def pad_spatial_axes(
    volume: npt.NDArray[np.float32], *, padding: Padding, mode: PadMode, value: float
) -> npt.NDArray[np.float32]:
    """Pad the spatial axes of a (C, Z, Y, X) volume, returning it as is when nothing is padded."""
    if all(axis_padding == (0, 0) for axis_padding in padding):
        return volume
    if mode == "constant":
        return np.pad(volume, pad_width=((0, 0), *padding), mode=mode, constant_values=value)
    return np.pad(volume, pad_width=((0, 0), *padding), mode=mode)
