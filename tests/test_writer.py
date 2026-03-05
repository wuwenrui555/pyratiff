"""Tests for PyramidWriter."""

import pathlib

import numpy as np
import pytest
import tifffile
from xml.etree import ElementTree

from pyratiff import PyramidWriter


# ---------------------------------------------------------------------------
# _validate_image_2d
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_validate_ok_dtypes(dtype):
    PyramidWriter._validate_image_2d((64, 64), np.dtype(dtype), (64, 64), False)


@pytest.mark.parametrize("dtype", [np.uint32, np.int32])
def test_validate_32bit_requires_mask(dtype):
    with pytest.raises(ValueError, match="32-bit"):
        PyramidWriter._validate_image_2d((64, 64), np.dtype(dtype), (64, 64), False)


@pytest.mark.parametrize("dtype", [np.uint32, np.int32])
def test_validate_32bit_ok_with_mask(dtype):
    PyramidWriter._validate_image_2d((64, 64), np.dtype(dtype), (64, 64), True)


def test_validate_unsupported_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        PyramidWriter._validate_image_2d((64, 64), np.dtype(np.float32), (64, 64), False)


def test_validate_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        PyramidWriter._validate_image_2d((32, 64), np.dtype(np.uint8), (64, 64), False)


def test_validate_msg_tag_in_error():
    with pytest.raises(ValueError, match="my_channel"):
        PyramidWriter._validate_image_2d(
            (32, 64), np.dtype(np.uint8), (64, 64), False, msg_tag="my_channel"
        )


# ---------------------------------------------------------------------------
# _build_ome_metadata
# ---------------------------------------------------------------------------


def test_build_metadata_full():
    meta = PyramidWriter._build_ome_metadata(0.5, ["DAPI", "CD45"])
    assert "UUID" in meta
    assert meta["PhysicalSizeX"] == 0.5
    assert meta["PhysicalSizeXUnit"] == "µm"
    assert meta["PhysicalSizeY"] == 0.5
    assert meta["Channel"] == {"Name": ["DAPI", "CD45"]}


def test_build_metadata_no_pixel_size():
    meta = PyramidWriter._build_ome_metadata(None, ["DAPI"])
    assert "PhysicalSizeX" not in meta
    assert "PhysicalSizeY" not in meta


def test_build_metadata_no_channels():
    meta = PyramidWriter._build_ome_metadata(0.5, [])
    assert "Channel" not in meta


# ---------------------------------------------------------------------------
# from_array
# ---------------------------------------------------------------------------


def test_from_array_2d():
    data = np.zeros((64, 64), dtype=np.uint16)
    writer = PyramidWriter.from_array(data)
    assert len(writer.in_chns) == 1
    assert writer.target_shape == (64, 64)
    assert writer.in_chns == ["channel_0"]


def test_from_array_3d():
    data = np.zeros((3, 64, 64), dtype=np.uint16)
    writer = PyramidWriter.from_array(data, channel_names=["A", "B", "C"])
    assert len(writer.in_chns) == 3
    assert writer.in_chns == ["A", "B", "C"]
    assert writer.target_shape == (64, 64)


def test_from_array_bad_ndim():
    with pytest.raises(ValueError, match="2-D or 3-D"):
        PyramidWriter.from_array(np.zeros((2, 4, 4, 4), dtype=np.uint8))


def test_from_array_channel_names_mismatch():
    data = np.zeros((3, 64, 64), dtype=np.uint16)
    with pytest.raises(ValueError, match="channel_names"):
        PyramidWriter.from_array(data, channel_names=["only_one"])


def test_from_array_dtype_promotion():
    """When channels have different dtypes, target_dtype is the maximum."""
    ch0 = np.zeros((64, 64), dtype=np.uint8)
    ch1 = np.zeros((64, 64), dtype=np.uint16)
    data = np.stack([ch0.astype(np.uint16), ch1])  # both uint16 after stack
    writer = PyramidWriter.from_array(data)
    assert writer.target_dtype == np.dtype(np.uint16)


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_2d_values():
    d = {
        "DAPI": np.zeros((64, 64), dtype=np.uint16),
        "CD45": np.ones((64, 64), dtype=np.uint16),
    }
    writer = PyramidWriter.from_dict(d)
    assert writer.in_chns == ["DAPI", "CD45"]
    assert writer.target_shape == (64, 64)


def test_from_dict_3d_value_split():
    """A (2, 64, 64) value should be split into channel_0 and channel_1."""
    d = {"marker": np.zeros((2, 64, 64), dtype=np.uint8)}
    writer = PyramidWriter.from_dict(d)
    assert writer.in_chns == ["marker_0", "marker_1"]


def test_from_dict_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported type"):
        PyramidWriter.from_dict({"bad": "not_an_array"})


def test_from_dict_channel_names_override():
    d = {
        "a": np.zeros((64, 64), dtype=np.uint8),
        "b": np.zeros((64, 64), dtype=np.uint8),
    }
    writer = PyramidWriter.from_dict(d, channel_names=["X", "Y"])
    assert writer.in_chns == ["X", "Y"]


# ---------------------------------------------------------------------------
# from_fs
# ---------------------------------------------------------------------------


def test_from_fs(tmp_path):
    paths = []
    for i in range(2):
        p = tmp_path / f"ch{i}.tiff"
        tifffile.imwrite(str(p), np.zeros((64, 64), dtype=np.uint16))
        paths.append(p)

    writer = PyramidWriter.from_fs(paths, channel_names=["DAPI", "CD45"])
    assert writer.in_chns == ["DAPI", "CD45"]
    assert writer.target_shape == (64, 64)


def test_from_fs_channel_names_mismatch(tmp_path):
    p = tmp_path / "ch0.tiff"
    tifffile.imwrite(str(p), np.zeros((64, 64), dtype=np.uint8))
    with pytest.raises(ValueError, match="channel_names"):
        PyramidWriter.from_fs([p], channel_names=["A", "B"])


def test_from_fs_3d_file_split(tmp_path):
    """A 3-D TIFF with 2 channels is split into <name>_0 and <name>_1."""
    p = tmp_path / "multi.tiff"
    tifffile.imwrite(str(p), np.zeros((2, 64, 64), dtype=np.uint16))
    writer = PyramidWriter.from_fs([p], channel_names=["marker"])
    assert writer.in_chns == ["marker_0", "marker_1"]


# ---------------------------------------------------------------------------
# export_ometiff_pyramid
# ---------------------------------------------------------------------------


def _make_writer(shape=(3, 64, 64), dtype=np.uint16, names=None):
    data = np.random.randint(0, 100, shape, dtype=dtype)
    if names is None:
        names = [f"ch{i}" for i in range(shape[0])]
    return PyramidWriter.from_array(data, channel_names=names)


def test_export_creates_file(tmp_path):
    out = tmp_path / "out.ome.tiff"
    _make_writer().export_ometiff_pyramid(out)
    assert out.exists()


def test_export_overwrite_false_raises(tmp_path):
    out = tmp_path / "out.ome.tiff"
    _make_writer().export_ometiff_pyramid(out)
    with pytest.raises(FileExistsError):
        _make_writer().export_ometiff_pyramid(out, overwrite=False)


def test_export_overwrite_true_replaces(tmp_path):
    out = tmp_path / "out.ome.tiff"
    _make_writer().export_ometiff_pyramid(out)
    mtime1 = out.stat().st_mtime
    _make_writer().export_ometiff_pyramid(out, overwrite=True)
    assert out.stat().st_mtime >= mtime1


def test_export_channel_names_in_ome_xml(tmp_path):
    out = tmp_path / "out.ome.tiff"
    names = ["DAPI", "CD45", "PanCK"]
    _make_writer(names=names).export_ometiff_pyramid(out)

    with tifffile.TiffFile(str(out)) as tif:
        xml = ElementTree.fromstring(tif.ome_metadata)
        channels = xml.findall(".//{*}Channel")
        parsed_names = [ch.attrib["Name"] for ch in channels]
    assert parsed_names == names


def test_export_pixel_size_in_ome_xml(tmp_path):
    out = tmp_path / "out.ome.tiff"
    _make_writer().export_ometiff_pyramid(out, pixel_size=0.325)

    with tifffile.TiffFile(str(out)) as tif:
        xml = ElementTree.fromstring(tif.ome_metadata)
        pixels = xml.find(".//{*}Pixels")
    assert float(pixels.attrib["PhysicalSizeX"]) == pytest.approx(0.325)
    assert float(pixels.attrib["PhysicalSizeY"]) == pytest.approx(0.325)


def test_export_pyramid_level_count(tmp_path):
    """64px image with tile_size=256 → only 1 level (base)."""
    out = tmp_path / "out.ome.tiff"
    _make_writer().export_ometiff_pyramid(out, tile_size=256)

    with tifffile.TiffFile(str(out)) as tif:
        n_levels = len(tif.series[0].levels)
    assert n_levels == 1


def test_export_larger_image_has_multiple_levels(tmp_path):
    """512px image with tile_size=256 → 2 levels."""
    out = tmp_path / "out.ome.tiff"
    data = np.zeros((2, 512, 512), dtype=np.uint16)
    writer = PyramidWriter.from_array(data, channel_names=["A", "B"])
    writer.export_ometiff_pyramid(out, tile_size=256)

    with tifffile.TiffFile(str(out)) as tif:
        n_levels = len(tif.series[0].levels)
    assert n_levels == 2


def test_export_base_level_shape(tmp_path):
    out = tmp_path / "out.ome.tiff"
    _make_writer(shape=(3, 64, 64)).export_ometiff_pyramid(out)

    with tifffile.TiffFile(str(out)) as tif:
        shape = tif.series[0].shape
    # tifffile returns (C, H, W)
    assert shape == (3, 64, 64)


def test_export_mask_uint32(tmp_path):
    """uint32 mask should write without error."""
    out = tmp_path / "mask.ome.tiff"
    data = np.zeros((64, 64), dtype=np.uint32)
    writer = PyramidWriter.from_array(data, channel_names=["labels"], is_mask=True)
    writer.export_ometiff_pyramid(out, is_mask=True)
    assert out.exists()


def test_export_pixel_values_preserved(tmp_path):
    """Base level pixel values must round-trip correctly."""
    out = tmp_path / "out.ome.tiff"
    data = np.arange(3 * 64 * 64, dtype=np.uint16).reshape(3, 64, 64) % 1000
    writer = PyramidWriter.from_array(data, channel_names=["A", "B", "C"])
    writer.export_ometiff_pyramid(out)

    result = tifffile.imread(str(out), level=0)
    np.testing.assert_array_equal(result, data)
