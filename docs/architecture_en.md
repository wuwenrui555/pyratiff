# pyratiff — Architecture

## Overview

pyratiff provides two public classes for working with pyramidal OME-TIFF files:

| Class | Purpose |
|---|---|
| `TiffZarrReader` | Lazy, channel-indexed reader backed by zarr |
| `PyramidWriter` | Multi-level OME-TIFF writer with tile-based compression |

Both are exposed directly from the top-level package:

```python
from pyratiff import TiffZarrReader, PyramidWriter
```

---

## File structure

```
src/pyratiff/
├── __init__.py   exposes TiffZarrReader and PyramidWriter
├── reader.py     TiffZarrReader
└── writer.py     PyramidWriter
```

---

## TiffZarrReader (`reader.py`)

### Design goal

Open a multi-channel TIFF without loading pixel data into memory. Expose
a full-image zarr array and per-channel zarr arrays so callers can slice
arbitrary regions lazily.

### Construction

Three entry points are provided. All ultimately call `__init__`, which opens
two zarr handles per file:

| Entry point | How channel names are obtained |
|---|---|
| `TiffZarrReader(path, channel_names)` | Caller supplies names |
| `TiffZarrReader.from_ometiff(path)` | Parsed from OME-XML `<Channel Name="...">` |
| `TiffZarrReader.from_qptiff(path)` | Parsed from `ScanProfile` JSON in `ImageDescription` |

All three also accept a `markerlist_f` (plain-text, one name per line) that
overrides the auto-detected names.

### Zarr 3 compatibility

tifffile ≥ 2025.5.21 requires zarr ≥ 3. Key differences from zarr 2:

- `zarr.open(store, mode="r")` — `mode` must be explicit.
- `zarr_array.astype(dtype)` — removed; use `np.asarray(arr).astype(dtype)`.

### Duplicate channel name deduplication

If the supplied `channel_names` contain duplicates, `_deduplicate_names`
adds `_0`, `_1`, … suffixes to **all occurrences** of every repeated name
before `zimg_dict` is built. This prevents silent key collision in the dict.

```python
["DAPI", "CD45", "DAPI"]  →  ["DAPI_0", "CD45", "DAPI_1"]
```

Unique names are returned unchanged. Deduplication always runs in `__init__`
after the length check, so `self.channel_names` and `zimg_dict` are always
consistent.

### Key attributes

```
reader.zimg          zarr.Array, shape (C, H, W) or (H, W)
reader.channel_names list[str], one entry per channel (deduplicated)
reader.zimg_dict     dict[str, zarr.Array], each (H, W)
```

### Helper methods

| Method | Returns |
|---|---|
| `channel_index(name)` | `int` index in `channel_names` |
| `channel_index([name, ...])` | `list[int]` |
| `slice_array(ymin, ymax, xmin, xmax)` | `np.ndarray (C, H, W)` |
| `slice_dict(ymin, ymax, xmin, xmax)` | `dict[str, np.ndarray]` each `(H, W)` |

### Channel-name extraction

**OME-TIFF** — reads the OME-XML string from `tif.ome_metadata` and finds
all `{*}Channel` elements (namespace-agnostic). Falls back to
`["Channel 0", ...]` when no `Name` attribute is present.

**QPTIFF** — reads the `ImageDescription` TIFF tag, parses the embedded
`<ScanProfile>` JSON, filters rows where:
- `markerName != "--"` (non-background)
- `id` does not match `^0+(-0+)+$` (non-empty-channel sentinel)

then deduplicates on `(id, markerName)`.

---

## PyramidWriter (`writer.py`)

### Design goal

Write a tiled, multi-level pyramidal OME-TIFF from in-memory arrays or from
single-channel TIFF files on disk. The pyramid is built by reading the
partially-written output file at each level and downsampling 2× — the same
strategy used by
[LabSysPharm/ome-tiff-pyramid-tools](https://github.com/labsyspharm/ome-tiff-pyramid-tools).

### Construction

Three class methods accept different input formats and normalise them into
the internal representation — a flat list of 2-D `np.ndarray` per channel:

| Entry point | Input |
|---|---|
| `PyramidWriter.from_array(arr)` | `(H, W)` or `(C, H, W)` array |
| `PyramidWriter.from_dict(dict)` | `{name: array}` mapping |
| `PyramidWriter.from_fs([paths])` | list of TIFF file paths |

All three accept `channel_names` (override) and `is_mask` (allow 32-bit
dtypes, switch to nearest-neighbour downsampling).

3-D inputs `(C, H, W)` in `from_dict` and `from_fs` are split into
individual channels named `<base>_0`, `<base>_1`, …

### Dtype validation

| dtype | Regular image | Mask (`is_mask=True`) |
|---|---|---|
| uint8 | ✓ | ✓ |
| uint16 | ✓ | ✓ |
| float32 | ✓ | ✓ |
| float64 | ✓ | ✓ |
| uint32 / int32 | ✗ | ✓ |
| other | ✗ | ✗ |

All channels are cast to the maximum dtype among the inputs before writing.

### Pyramid geometry

```python
num_levels = max(1, ceil(log2(max(H, W) / tile_size)) + 1)
factors    = 2 ** arange(num_levels)              # [1, 2, 4, ...]
shapes     = ceil([H, W] / factors[:, None])      # shape at each level
cshapes    = ceil(shapes / tile_size)             # tile counts at each level
```

### Tile generation

Two generator functions are created by `_make_tile_generators`:

**`tiles0()`** — streams tiles for level 0 directly from `self.in_imgs`
(already in memory), channel by channel, row by row.

**`tiles(level)`** — opens the partially-written output file, reads tiles
from `level - 1` via zarr, downsamples by 2× using a thread pool, and
yields the results. Downsampling strategy:

- **Regular images** — `skimage.transform.downscale_local_mean`, then
  `np.round().astype(dtype)` for integer dtypes, or `.astype(dtype)` for
  float dtypes (rounding is skipped to preserve precision).
- **Masks** (`is_mask=True`) — strided indexing `[::2, ::2]` to preserve
  integer label values without interpolation.

### OME-XML metadata

Written via tifffile's `metadata` dict:

```python
{
    "UUID": uuid.uuid4().urn,
    "PhysicalSizeX": pixel_size, "PhysicalSizeXUnit": "µm",
    "PhysicalSizeY": pixel_size, "PhysicalSizeYUnit": "µm",
    "Channel": {"Name": channel_names},
}
```

`pixel_size` and `channel_names` are omitted when `None` / empty.

### Compression and predictor

All levels use `compression="adobe_deflate"` with `predictor=True`.
tifffile selects the appropriate predictor automatically:

- Integer dtypes → horizontal differencing predictor (built into tifffile)
- Float dtypes → floating-point predictor (PREDICTOR type 3, requires
  `imagecodecs`)

### Threading

`tifffile.TIFF.MAXWORKERS` and `tifffile.TIFF.MAXIOWORKERS` are set before
writing to control internal tifffile threads. The downsampling pool uses
`concurrent.futures.ThreadPoolExecutor` with `num_threads` workers.

Pass `num_threads=0` to auto-detect from CPU affinity.

### Writing flow

```
for level in 0 .. num_levels-1:
    if level == 0:
        writer.write(data=tiles0(), subifds=num_levels-1, ...)  # base level
    else:
        writer.write(data=tiles(level), subfiletype=1, ...)     # reduced image
```

`subifds` pre-allocates IFD slots for all pyramid levels within the same
TIFF strip, which is required by the OME-TIFF pyramid specification.
`subfiletype=1` marks reduced-resolution images per the TIFF spec.

---

## Dependencies

| Package | Minimum version | Why |
|---|---|---|
| tifffile | 2025.5.21 | zarr 3 support |
| zarr | 3.0 | new API (zarr 2 removed) |
| numpy | 1.21 | array operations |
| scikit-image | 0.19 | `downscale_local_mean` |
| pandas | 1.5 | metadata parsing |
| tqdm | 4.0 | progress bars |
| imagecodecs | — | floating-point predictor for float32/float64 TIFF |
