# pyratiff

Read and write pyramidal OME-TIFF files.

## Installation

```bash
pip install git+https://github.com/wuwenrui555/pyratiff.git
```

Or with uv:

```bash
uv add git+https://github.com/wuwenrui555/pyratiff.git
```

## Requirements

- Python >= 3.12
- tifffile >= 2025.5.21 (requires zarr >= 3)
- zarr >= 3.0
- imagecodecs (required for float32/float64 predictor compression)

## Quick start

### Reading

```python
from pyratiff import TiffZarrReader

# From OME-TIFF — channel names are read from OME-XML metadata
reader = TiffZarrReader.from_ometiff("image.ome.tiff")
print(reader.channel_names)

# From QPTIFF — channel names are parsed from the ScanProfile JSON
reader = TiffZarrReader.from_qptiff("image.qptiff")

# Override channel names from a plain-text file (one name per line)
reader = TiffZarrReader.from_ometiff("image.ome.tiff", markerlist_f="markers.txt")

# Access the full image as a zarr array (C, H, W) — lazy, no data loaded yet
img = reader.zimg

# Select channels by name and crop a region
indices = reader.channel_index(["DAPI", "CD45"])
patch = reader.zimg[indices, :512, :512]

# Crop as a dict of (H, W) numpy arrays, keyed by channel name
region = reader.slice_dict(ymin=0, ymax=512, xmin=0, xmax=512)
```

> **Duplicate channel names**: if the file contains channels with the same
> name, suffixes `_0`, `_1`, … are appended automatically so that
> `zimg_dict` keys are always unique.

### Writing

```python
import numpy as np
from pyratiff import PyramidWriter

# From a (C, H, W) numpy array — uint8, uint16, float32, float64 supported
data = np.random.randint(0, 65535, (4, 8192, 8192), dtype=np.uint16)
channel_names = ["DAPI", "CD45", "PanCK", "CD8"]

writer = PyramidWriter.from_array(data, channel_names=channel_names)
writer.export_ometiff_pyramid(
    "output.ome.tiff",
    pixel_size=0.5,    # µm, written to OME-XML PhysicalSizeX/Y
    tile_size=256,
    num_threads=8,
)

# From a list of single-channel TIFF files
writer = PyramidWriter.from_fs(
    ["dapi.tiff", "cd45.tiff"],
    channel_names=["DAPI", "CD45"],
)
writer.export_ometiff_pyramid("output.ome.tiff")

# From a dict of arrays
writer = PyramidWriter.from_dict({"DAPI": dapi_array, "CD45": cd45_array})
writer.export_ometiff_pyramid("output.ome.tiff")

# Label mask — allows uint32/int32 and uses nearest-neighbour downsampling
mask = np.zeros((8192, 8192), dtype=np.uint32)
writer = PyramidWriter.from_array(mask, channel_names=["labels"], is_mask=True)
writer.export_ometiff_pyramid("mask.ome.tiff", is_mask=True)
```

## API reference

### `TiffZarrReader`

| | |
|---|---|
| `TiffZarrReader(path, channel_names=None)` | Open any TIFF with manually supplied names |
| `TiffZarrReader.from_ometiff(path, markerlist_f=None)` | Names from OME-XML |
| `TiffZarrReader.from_qptiff(path, markerlist_f=None)` | Names from ScanProfile JSON |

| Attribute | Type | Description |
|---|---|---|
| `zimg` | `zarr.Array` | Full image, shape `(C, H, W)` or `(H, W)` |
| `channel_names` | `list[str]` | Channel names (deduplicated if needed) |
| `zimg_dict` | `dict[str, zarr.Array]` | Per-channel `(H, W)` lazy arrays |

| Method | Returns |
|---|---|
| `channel_index(name)` | `int` |
| `channel_index([name, ...])` | `list[int]` |
| `slice_array(ymin, ymax, xmin, xmax)` | `np.ndarray (C, H, W)` |
| `slice_dict(ymin, ymax, xmin, xmax)` | `dict[str, np.ndarray]` |

### `PyramidWriter`

| Constructor | Input |
|---|---|
| `PyramidWriter.from_array(arr, channel_names=None, is_mask=False)` | `(H, W)` or `(C, H, W)` array |
| `PyramidWriter.from_dict(dict, channel_names=None, is_mask=False)` | `{name: array}` |
| `PyramidWriter.from_fs(paths, channel_names=None, is_mask=False)` | list of TIFF paths |

**Supported dtypes**

| dtype | Regular image | Mask (`is_mask=True`) |
|---|---|---|
| uint8 | ✓ | ✓ |
| uint16 | ✓ | ✓ |
| uint32 | ✓ | ✓ |
| float32 | ✓ | ✗ |
| float64 | ✓ | ✗ |

`export_ometiff_pyramid(output_f, pixel_size=None, tile_size=256, is_mask=False, num_threads=8, overwrite=True)`

## Documentation

See [`docs/architecture_en.md`](docs/architecture_en.md) for internal design details.

## License

Apache-2.0
