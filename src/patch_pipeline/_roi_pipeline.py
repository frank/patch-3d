"""The RoiPipeline class, serving one high-resolution patch over a region of interest."""

from collections.abc import Iterator, Sequence

import numpy as np
import numpy.typing as npt

from patch_pipeline._geometry import (
    PerAxis,
    Shape,
    broadcast_per_axis,
    effective_spacing,
    non_negative_float,
    optional,
    positive_float,
    positive_int,
)
from patch_pipeline._pipeline import (
    AxisArgument,
    PatchIndex,
    PatchPipeline,
    channel_first,
    numeric_array,
    output_array,
    spatial_shape,
)
from patch_pipeline._roi import Region, channel_fill, place_region, region_geometry, roi_weights


class RoiPipeline:
    """Serve one high-resolution patch over a region of interest, and return its output on the
    original voxel grid.

    A low-resolution `roi_mask`, typically from a coarse `PatchPipeline` pass, decides where the
    patch goes: at the position covering the most mask weight, and among tied positions the one
    closest to the mask centroid. Construction crops the volume to the original voxels that patch
    spans, its `region`, and hands the crop to an inner `PatchPipeline`, so neither the full
    volume at high resolution nor the caller's arrays are kept. Patches are fetched and fed as
    with `PatchPipeline`. `finalize` returns the full original shape, with `fill_value` outside
    the region, and `finalize_region` returns the region only.

    Conventions:

    - Axis order: every per-axis argument follows the axis order of the numpy array, as in
      `PatchPipeline`.
    - Mask extent: `roi_mask` must cover the same physical extent as `volume`, at any shape, so
      that mask position e maps to volume position e * N0 / M in edge coordinates. This holds
      for the `working_output()` of a `PatchPipeline` built on the same volume, and cannot be
      checked.
    - Region: it spans patch_size * target_spacing / source_spacing original voxels, rounded,
      on axes that are resampled, and patch_size elsewhere, independent of the volume length.
      An axis is resampled when it would be in `PatchPipeline`, and also when only the
      rounding of the whole volume would leave it unchanged. On an axis shorter than the
      region, the region is the whole axis, worked at the length `PatchPipeline` would give it
      and padded to the patch.
    - Deviations: the patch grid may be offset from a whole-volume working grid by a fraction
      of a voxel, with spacing `effective_spacing` rather than exactly `target_spacing`. When
      upsampling, working voxels at the region border extrapolate within half a voxel instead
      of seeing the data just outside it, and `antialias_in` sees the crop edge. The output
      has a hard edge where `fill_value` takes over.
    - Concurrency: as in `PatchPipeline`.

    Args:
        volume: (Z, Y, X), or channel-first (C_in, Z, Y, X), of integer or floating dtype, on
            the original voxel grid.
        roi_mask: (Z_m, Y_m, X_m) non-negative weights of bool, integer or floating dtype,
            with at least one positive voxel. Its reference is dropped after placement.
        patch_size: Patch size per axis, or one size for all axes.
        source_spacing: Voxel spacing of `volume`, needed only on axes with a target spacing.
        target_spacing: Patch spacing per axis. Axes without one are not resampled.
        spacing_rtol: Relative tolerance per axis on the target spacing, as in `PatchPipeline`.
        interp_in, interp_out, antialias_in, antialias_out, pad_mode, pad_value, num_threads:
            As in `PatchPipeline`, applied to the region only.

    Raises:
        TypeError, ValueError: An argument is invalid, or `roi_mask` has no positive voxels.
    """

    def __init__(
        self,
        volume: npt.ArrayLike,
        *,
        roi_mask: npt.ArrayLike,
        patch_size: AxisArgument[int],
        source_spacing: AxisArgument[float | None] = None,
        target_spacing: AxisArgument[float | None] = None,
        spacing_rtol: AxisArgument[float | None] = None,
        interp_in: str = "linear",
        interp_out: str = "linear",
        antialias_in: bool = False,
        antialias_out: bool = False,
        pad_mode: str = "constant",
        pad_value: float = 0.0,
        num_threads: int | None = None,
    ) -> None:
        volume = channel_first(numeric_array(volume, name="volume"))
        self._original_shape = spatial_shape(volume)
        checked_patch_size = broadcast_per_axis(patch_size, name="patch_size", convert=positive_int)
        known_spacing = broadcast_per_axis(
            source_spacing, name="source_spacing", convert=optional(positive_float)
        )
        region_shape, working_shape = region_geometry(
            original_shape=self._original_shape,
            patch_size=checked_patch_size,
            source_spacing=known_spacing,
            target_spacing=broadcast_per_axis(
                target_spacing, name="target_spacing", convert=optional(positive_float)
            ),
            spacing_rtol=broadcast_per_axis(
                spacing_rtol, name="spacing_rtol", convert=optional(non_negative_float)
            ),
        )
        self._effective_spacing = effective_spacing(
            input_shape=region_shape, working_shape=working_shape, source_spacing=known_spacing
        )
        self._region, self._coverage = place_region(
            roi_weights(roi_mask, name="roi_mask"),
            original_shape=self._original_shape,
            region_shape=region_shape,
        )
        # the inner pipeline copies the crop, so no reference to the caller's volume is kept
        self._inner = PatchPipeline(
            volume[:, *self._region],
            patch_size=checked_patch_size,
            target_shape=working_shape,
            interp_in=interp_in,
            interp_out=interp_out,
            antialias_in=antialias_in,
            antialias_out=antialias_out,
            pad_mode=pad_mode,
            pad_value=pad_value,
            num_threads=num_threads,
        )

    @property
    def original_shape(self) -> Shape:
        """The spatial shape of the volume, which `finalize` returns."""
        return self._original_shape

    @property
    def region(self) -> Region:
        """Where the patch lies in the original grid, as slices per axis."""
        return self._region

    @property
    def coverage(self) -> float:
        """Fraction of the mask weight inside the region, at mask resolution; 1.0 if it all
        fits."""
        return self._coverage

    @property
    def effective_spacing(self) -> PerAxis[float | None]:
        """The patch spacing source_spacing * region_length / working_length, or None where the
        source spacing is unknown."""
        return self._effective_spacing

    @property
    def working_shape(self) -> Shape:
        """The region's resampled shape, before padding."""
        return self._inner.working_shape

    @property
    def patch_size(self) -> Shape:
        return self._inner.patch_size

    @property
    def grid_shape(self) -> Shape:
        """See `PatchPipeline.grid_shape`."""
        return self._inner.grid_shape

    @property
    def locations(self) -> npt.NDArray[np.intp]:
        """Start voxel of each patch in the region's padded working coordinates."""
        return self._inner.locations

    @property
    def fed(self) -> npt.NDArray[np.bool_]:
        """See `PatchPipeline.fed`."""
        return self._inner.fed

    @property
    def is_complete(self) -> bool:
        return self._inner.is_complete

    @property
    def n_channels_in(self) -> int:
        return self._inner.n_channels_in

    @property
    def n_channels_out(self) -> int | None:
        """See `PatchPipeline.n_channels_out`."""
        return self._inner.n_channels_out

    def __len__(self) -> int:
        return len(self._inner)

    def __getitem__(self, idx: PatchIndex) -> npt.NDArray[np.float32]:
        """See `PatchPipeline.__getitem__`."""
        return self._inner[idx]

    def feed(self, idx: PatchIndex, outputs: npt.ArrayLike) -> None:
        """See `PatchPipeline.feed`."""
        self._inner.feed(idx, outputs)

    def __iter__(self) -> Iterator[tuple[int, npt.NDArray[np.float32]]]:
        return iter(self._inner)

    def batches(
        self, batch_size: int
    ) -> Iterator[tuple[npt.NDArray[np.intp], npt.NDArray[np.float32]]]:
        """See `PatchPipeline.batches`."""
        return self._inner.batches(batch_size)

    def working_output(self) -> npt.NDArray[np.float32]:
        """The normalized output in the region's working domain, (C_out, *working_shape). See
        `PatchPipeline.working_output`."""
        return self._inner.working_output()

    def finalize_region(
        self, out: npt.NDArray[np.float32] | None = None
    ) -> npt.NDArray[np.float32]:
        """The output resampled onto the region, (C_out, *region_shape), float32.

        It is written into `out` if given, which may be a view such as
        `my_volume[:, *pipeline.region]`. Finalizing releases the accumulator, so either
        finalize method can be called only once.

        Raises:
            RuntimeError: Some patches have not been fed, or finalizing was already done.
            TypeError, ValueError: `out` is not a float32 array of the region's output shape.
        """
        working = self._inner.working_output()
        out = output_array(
            out,
            shape=(len(working), *self._inner.original_shape),
            layout="(C_out, *region_shape)",
        )
        return self._inner.finalize(out=out)

    def finalize(
        self,
        out: npt.NDArray[np.float32] | None = None,
        fill_value: float | Sequence[float] = 0.0,
    ) -> npt.NDArray[np.float32]:
        """The output on the original grid, (C_out, *original_shape), float32, with
        `fill_value` everywhere outside the region.

        `fill_value` is a scalar for every channel or one finite value per output channel; for
        softmax outputs, mark the background explicitly, e.g. (1.0, 0.0, ...). `out` may be an
        np.memmap. The whole call is validated before anything is written, so a rejected call
        leaves `out` and the pipeline unchanged. Either finalize method can be called only once.

        Raises:
            RuntimeError: Some patches have not been fed, or finalizing was already done.
            TypeError, ValueError: `fill_value` or `out` is invalid.
        """
        # raises before anything is written if patches are missing or finalizing was done
        n_channels_out = len(self._inner.working_output())
        fill = channel_fill(fill_value, n_channels=n_channels_out)
        shape = (n_channels_out, *self._original_shape)
        if out is None and not fill.any():
            # pages outside the region stay unallocated until written
            out = np.zeros(shape, dtype=np.float32)
        else:
            out = output_array(out, shape=shape, layout="(C_out, *original_shape)")
            out[...] = fill[:, None, None, None]
        self.finalize_region(out=out[:, *self._region])
        return out

    def __repr__(self) -> str:
        region = ", ".join(f"{axis.start}:{axis.stop}" for axis in self._region)
        return (
            f"{type(self).__name__}("
            f"original_shape={self.original_shape}, region=({region}), "
            f"working_shape={self.working_shape}, patch_size={self.patch_size}, "
            f"coverage={self.coverage:.3f}, n_channels_in={self.n_channels_in}, "
            f"n_channels_out={self.n_channels_out}, "
            f"fed={np.count_nonzero(self._inner.fed)}/{len(self)})"
        )
