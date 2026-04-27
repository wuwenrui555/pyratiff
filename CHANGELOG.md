# Changelog

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
