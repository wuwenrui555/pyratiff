# Changelog

## [2.0.1] - 2026-04-26

### Changed

- **Richer omero metadata in OME-Zarr output.** The `omero` block now
  includes per-channel `color`, `active`, and `window` (alongside the
  existing `label`), plus omero-level `name`, `version`, and `rdefs`.
  napari-ome-zarr (and OMERO / IDR / IJ Bio-Formats) read these to render
  each channel as a separate Image layer with a sensible colormap and
  contrast — without them, viewers fall back to a single layer with a
  channel slider.
- Default channel colors cycle through a 7-color palette
  (white, red, green, blue, yellow, magenta, cyan); first channel is white
  (nucleus-friendly).
- Default `window` is the dtype's full range (`iinfo.min`..`iinfo.max` for
  ints, `0.0..1.0` for floats); viewers can auto-contrast on top of this.

### Why

Even though the v2.0.0 omero block was a valid (minimal) OME-NGFF v0.4
document, napari-ome-zarr needs more fields to render channels well.

## [2.0.0] - 2026-04-26

OME-Zarr is now the canonical pyramid format; OME-TIFF is derived from it.
Pipeline is fully lazy + streaming so larger-than-RAM images go through
without OOM. Output is byte-identical to v1.1.0 for in-memory inputs
(verified by the same golden-master and cross-format tests).

### Architecture changes

- **Input is lazy**. ``self.in_imgs`` now stores zarr-backed arrays (or
  numpy slices that share memory with the original) rather than eager
  numpy copies. Inputs from disk (``from_fs``, ``from_ome_zarr``) never load
  until pyramid generation reads each tile.
- **Pyramid is streamed tile-by-tile** in ``_stream_pyramid_to_zarr_group``.
  Memory peak is ``O(num_channels * tile_size² * dtype_size)`` per active
  tile, not ``O(full pyramid)``.
- **OME-Zarr is canonical**. ``export_ome_zarr`` writes a full OME-NGFF v0.4
  group via the streaming builder. ``export_ometiff_pyramid`` first writes a
  temporary OME-Zarr (in a system tempdir) and then derives the OME-TIFF
  from it via ``_zarr_group_to_ometiff``. The temp zarr is cleaned up
  automatically.

### Added

- **`PyramidWriter.from_ome_zarr(path)`** — lazy reader for an existing
  OME-NGFF v0.4 zarr group. Channel names come from the omero metadata
  when available, otherwise fall back to ``channel_0..N-1``.
- 5 new tests cover round-tripping through ``from_ome_zarr``, lazy input
  semantics, byte-identity across tile sizes, and temp-zarr cleanup.

### Removed (internal)

- ``_build_pyramid_levels`` (numpy-cascade approach from v1.0.1) — replaced
  by ``_stream_pyramid_to_zarr_group``.
- ``_tile_generator`` for in-memory levels — no longer needed; tiles come
  from zarr arrays directly.

### Breaking changes

- ``self.in_imgs`` element type changed from ``list[numpy.ndarray]`` to
  ``list[ArrayLike]`` (numpy arrays for ``from_array`` / ``from_dict`` with
  numpy values; zarr arrays or ``_ChannelView`` wrappers for ``from_fs``,
  ``from_dict`` with zarr values, and ``from_ome_zarr``). Subclasses or
  external code that introspect ``in_imgs`` may need to call
  ``np.asarray(item)`` before assuming numpy semantics.

### Why

Long-term direction is napari + OME-NGFF; the v2.0 architecture makes
OME-Zarr first-class and removes the eager-numpy assumption that previously
capped pyratiff at images that fit in RAM. All existing public API
signatures and outputs are preserved.

## [1.1.0] - 2026-04-26

### Added

- **`PyramidWriter.export_ome_zarr(output_dir, ...)`** — write the same image
  stack as an [OME-NGFF v0.4](https://ngff.openmicroscopy.org/0.4/) zarr
  group. Output is a directory (conventionally `*.ome.zarr`) containing one
  zarr array per pyramid level plus `multiscales` and `omero` metadata in
  `.zattrs`. Interoperable with napari (via `napari-ome-zarr`) and any
  OME-NGFF v0.4 reader.
  - Reuses the same in-memory pyramid cascade as `export_ometiff_pyramid`,
    so OME-Zarr and OME-TIFF outputs hold byte-identical pyramid data
    (verified by `test_ome_zarr_and_ometiff_byte_identical`).
  - Mask mode (`is_mask=True`) uses nearest-neighbour subsampling and sets
    `multiscales[0].type = "nearest"`.
  - Pixel size in microns flows through to the per-axis `unit` and
    coordinate-transformation `scale` fields.
- 9 new tests covering directory structure, overwrite semantics, multiscales
  metadata correctness, omero channel labels, mask mode, and cross-format
  byte-identity with OME-TIFF.

### Why

Long-term direction is napari + OME-NGFF; OME-Zarr is the modern format the
OME consortium is steering toward. OME-TIFF stays as the default for
QuPath compatibility. Both methods coexist; pick whichever the downstream
tool needs (or write both side by side from one writer instance).

## [1.0.1] - 2026-04-26

### Changed

- **Pyramid generation refactored to in-memory cascade** (internal). Previous
  versions used the labsyspharm trick of reading partially-written pyramid
  levels back from the output file to produce the next level. The new
  approach computes each level by 2× downsampling the previous level held in
  memory. Output is byte-identical (verified by golden-master tests across
  uint16, uint32 mask, and float32 inputs); behavior depends only on
  `skimage.transform.downscale_local_mean` rather than on any tifffile
  flush-on-write semantics.
- `print(...)` calls in `export_ometiff_pyramid` replaced with
  `logging.info(...)` (`pyratiff.writer` logger).
- `tifffile.TIFF.MAXWORKERS` and `MAXIOWORKERS` now saved and restored
  around each export, avoiding global-state pollution for other tifffile
  callers in the same process.
- `concurrent.futures.ThreadPoolExecutor` removed (no longer needed —
  per-tile work is just an array slice).

### Added

- 3 golden-master tests (`test_export_pyramid_level_values_uint16`,
  `..._mask`, `..._float`) verifying every pyramid level matches the
  reference downsample of its predecessor.

### Why

Removes the dependency on tifffile's undocumented "subifd flushed before
subsequent reads" behavior; simpler code path; easier to swap for a
zarr-native (OME-NGFF) backend in the future.

## [1.0.0] - 2026-03-05

Initial release.

### Features

- `TiffZarrReader` — lazy zarr-backed reader for OME-TIFF and QPTIFF files
  - Channel names auto-extracted from OME-XML (`from_ometiff`) or ScanProfile JSON (`from_qptiff`)
  - Override channel names from a plain-text marker list (`markerlist_f`)
  - Duplicate channel names automatically deduplicated with `_0`, `_1`, … suffixes to prevent `zimg_dict` key collision
  - Per-channel lazy access via `zimg_dict`; region crop via `slice_array` / `slice_dict`

- `PyramidWriter` — tiled pyramidal OME-TIFF writer
  - Three construction paths: `from_array`, `from_dict`, `from_fs`
  - Supported dtypes — regular images: uint8 / uint16 / uint32 / float32 / float64; masks: uint8 / uint16 / uint32
  - Mask mode (`is_mask=True`): nearest-neighbour downsampling to preserve integer label values
  - adobe_deflate compression with predictor for all dtypes (floating-point predictor via `imagecodecs`)
  - OME-XML metadata: channel names, physical pixel size
  - Multi-threaded tile generation via `concurrent.futures.ThreadPoolExecutor`

### Dependencies

- tifffile >= 2025.5.21 (zarr 3 support)
- zarr >= 3.0
- imagecodecs (floating-point predictor)
- numpy >= 1.21
- scikit-image >= 0.19
- pandas >= 1.5
- tqdm >= 4.0
