# Changelog

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
