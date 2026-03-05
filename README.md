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

## Quick start

### Reading

```python
from pyratiff import TiffZarrReader

# From OME-TIFF (channel names from OME-XML metadata)
reader = TiffZarrReader.from_ometiff("image.ome.tiff")
print(reader.channel_names)

# Access full image as zarr array (C, H, W) — lazy, no data loaded yet
img = reader.zimg

# Select channels and crop a region
indices = reader.channel_index(["DAPI", "CD45"])
patch = reader.zimg[indices, :512, :512]

# Per-channel dict of (H, W) zarr arrays
region = reader.slice_dict(ymin=0, ymax=512, xmin=0, xmax=512)
```

### Writing

```python
import numpy as np
from pyratiff import PyramidWriter

# From a (C, H, W) numpy array
data = np.random.randint(0, 65535, (4, 8192, 8192), dtype=np.uint16)
channel_names = ["DAPI", "CD45", "PanCK", "CD8"]

writer = PyramidWriter.from_array(data, channel_names=channel_names)
writer.export_ometiff_pyramid(
    "output.ome.tiff",
    pixel_size=0.5,   # microns
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
```

## Documentation

See [`docs/architecture_en.md`](docs/architecture_en.md) for a detailed description of the internal design.
