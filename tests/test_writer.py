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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.float32, np.float64])
def test_validate_ok_image_dtypes(dtype):
    PyramidWriter._validate_image_2d((64, 64), np.dtype(dtype), (64, 64), False)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
def test_validate_ok_mask_dtypes(dtype):
    PyramidWriter._validate_image_2d((64, 64), np.dtype(dtype), (64, 64), True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_validate_float_and_signed_rejected_as_mask(dtype):
    with pytest.raises(ValueError, match="Mask dtype must be unsigned integer"):
        PyramidWriter._validate_image_2d((64, 64), np.dtype(dtype), (64, 64), True)


def test_validate_unsupported_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        PyramidWriter._validate_image_2d((64, 64), np.dtype(np.int16), (64, 64), False)


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


def _expected_downsampled(prev: np.ndarray, is_mask: bool, target_dtype) -> np.ndarray:
    """Reference 2x downsample, mirrors PyramidWriter's per-tile logic."""
    import skimage.transform
    if is_mask:
        return prev[..., ::2, ::2].astype(target_dtype)
    if prev.ndim == 2:
        ds = skimage.transform.downscale_local_mean(prev, (2, 2))
    else:  # (C, H, W) → downsample H, W only
        ds = np.stack([
            skimage.transform.downscale_local_mean(prev[c], (2, 2))
            for c in range(prev.shape[0])
        ])
    if np.issubdtype(target_dtype, np.floating):
        return ds.astype(target_dtype)
    return np.round(ds).astype(target_dtype)


def test_export_pyramid_level_values_uint16(tmp_path):
    """Each pyramid level must be the local-mean downsample of the previous one (uint16)."""
    out = tmp_path / "out.ome.tiff"
    rng = np.random.default_rng(42)
    data = rng.integers(0, 1000, (2, 512, 512), dtype=np.uint16)
    writer = PyramidWriter.from_array(data, channel_names=["A", "B"])
    writer.export_ometiff_pyramid(out, tile_size=128)

    with tifffile.TiffFile(str(out)) as tif:
        n_levels = len(tif.series[0].levels)

    # Build expected pyramid via in-memory cascade.
    expected = [data]
    for _ in range(1, n_levels):
        expected.append(_expected_downsampled(expected[-1], is_mask=False,
                                              target_dtype=np.uint16))

    for L, exp in enumerate(expected):
        actual = tifffile.imread(str(out), level=L)
        np.testing.assert_array_equal(actual, exp, err_msg=f"level {L} mismatch")


def test_export_pyramid_level_values_mask(tmp_path):
    """Mask mode uses nearest-neighbour subsampling."""
    out = tmp_path / "mask.ome.tiff"
    rng = np.random.default_rng(7)
    # 2 channels so tifffile keeps the channel dim on read.
    data = rng.integers(0, 50, (2, 512, 512), dtype=np.uint16)
    writer = PyramidWriter.from_array(data, channel_names=["labels_a", "labels_b"], is_mask=True)
    writer.export_ometiff_pyramid(out, tile_size=128, is_mask=True)

    with tifffile.TiffFile(str(out)) as tif:
        n_levels = len(tif.series[0].levels)

    expected = [data]
    for _ in range(1, n_levels):
        expected.append(_expected_downsampled(expected[-1], is_mask=True,
                                              target_dtype=np.uint16))

    for L, exp in enumerate(expected):
        actual = tifffile.imread(str(out), level=L)
        np.testing.assert_array_equal(actual, exp, err_msg=f"level {L} mask mismatch")


def test_export_pyramid_level_values_float(tmp_path):
    """Float dtype skips the round() step in downsampling."""
    out = tmp_path / "out.ome.tiff"
    rng = np.random.default_rng(13)
    data = rng.random((2, 512, 512), dtype=np.float32)
    writer = PyramidWriter.from_array(data, channel_names=["A", "B"])
    writer.export_ometiff_pyramid(out, tile_size=128)

    with tifffile.TiffFile(str(out)) as tif:
        n_levels = len(tif.series[0].levels)

    expected = [data]
    for _ in range(1, n_levels):
        expected.append(_expected_downsampled(expected[-1], is_mask=False,
                                              target_dtype=np.float32))

    for L, exp in enumerate(expected):
        actual = tifffile.imread(str(out), level=L)
        np.testing.assert_allclose(actual, exp, rtol=1e-5,
                                   err_msg=f"level {L} float mismatch")


# ---------------------------------------------------------------------------
# float dtype support
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_from_array_float(dtype):
    data = np.random.rand(2, 64, 64).astype(dtype)
    writer = PyramidWriter.from_array(data, channel_names=["A", "B"])
    assert writer.target_dtype == np.dtype(dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_export_float_roundtrip(tmp_path, dtype):
    """Float images write and read back without dtype change."""
    out = tmp_path / "float.ome.tiff"
    data = np.random.rand(2, 64, 64).astype(dtype)
    writer = PyramidWriter.from_array(data, channel_names=["A", "B"])
    writer.export_ometiff_pyramid(out)

    result = tifffile.imread(str(out), level=0)
    assert result.dtype == np.dtype(dtype)
    np.testing.assert_allclose(result, data, rtol=1e-5)


# ---------------------------------------------------------------------------
# export_ome_zarr (OME-NGFF v0.4)
# ---------------------------------------------------------------------------

import zarr  # noqa: E402


def _open_zarr_group(path):
    return zarr.open_group(store=zarr.storage.LocalStore(str(path)), mode="r")


def test_ome_zarr_creates_directory(tmp_path):
    out = tmp_path / "out.ome.zarr"
    _make_writer().export_ome_zarr(out)
    assert out.is_dir()
    assert (out / ".zgroup").exists()
    assert (out / ".zattrs").exists()
    assert (out / "0" / ".zarray").exists()


def test_ome_zarr_overwrite_false_raises(tmp_path):
    out = tmp_path / "out.ome.zarr"
    _make_writer().export_ome_zarr(out)
    with pytest.raises(FileExistsError):
        _make_writer().export_ome_zarr(out, overwrite=False)


def test_ome_zarr_overwrite_true_replaces(tmp_path):
    out = tmp_path / "out.ome.zarr"
    _make_writer().export_ome_zarr(out)
    _make_writer().export_ome_zarr(out, overwrite=True)
    assert out.is_dir()


def test_ome_zarr_multiscales_metadata(tmp_path):
    out = tmp_path / "out.ome.zarr"
    _make_writer(shape=(2, 512, 512), names=["A", "B"]).export_ome_zarr(
        out, pixel_size=0.325, chunk_size=128
    )
    root = _open_zarr_group(out)
    ms = root.attrs["multiscales"][0]
    assert ms["version"] == "0.4"
    assert ms["axes"][0]["name"] == "c"
    assert ms["axes"][0]["type"] == "channel"
    assert ms["axes"][1]["name"] == "y"
    assert ms["axes"][1]["unit"] == "micrometer"
    assert ms["axes"][2]["name"] == "x"
    # 512 / 128 = 4 → ceil(log2(4)) + 1 = 3 levels
    assert len(ms["datasets"]) == 3
    assert ms["datasets"][0]["path"] == "0"
    assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 0.325, 0.325]
    assert ms["datasets"][1]["coordinateTransformations"][0]["scale"] == [1.0, 0.65, 0.65]
    assert ms["type"] == "mean"


def test_ome_zarr_no_pixel_size_omits_unit(tmp_path):
    out = tmp_path / "out.ome.zarr"
    _make_writer().export_ome_zarr(out)  # no pixel_size
    root = _open_zarr_group(out)
    ms = root.attrs["multiscales"][0]
    for axis in ms["axes"][1:]:  # spatial axes
        assert "unit" not in axis
    # scales are pure 2^L ratios
    assert ms["datasets"][0]["coordinateTransformations"][0]["scale"] == [1.0, 1.0, 1.0]


def test_ome_zarr_omero_channel_names(tmp_path):
    out = tmp_path / "out.ome.zarr"
    names = ["DAPI", "CD45", "PanCK"]
    _make_writer(names=names).export_ome_zarr(out)
    root = _open_zarr_group(out)
    omero = root.attrs["omero"]
    assert [c["label"] for c in omero["channels"]] == names


def test_ome_zarr_pyramid_level_values_uint16(tmp_path):
    """OME-Zarr levels should match the same in-memory cascade as OME-TIFF."""
    out = tmp_path / "out.ome.zarr"
    rng = np.random.default_rng(99)
    data = rng.integers(0, 1000, (2, 512, 512), dtype=np.uint16)
    writer = PyramidWriter.from_array(data, channel_names=["A", "B"])
    writer.export_ome_zarr(out, chunk_size=128)

    root = _open_zarr_group(out)
    n_levels = len(root.attrs["multiscales"][0]["datasets"])

    expected = [data]
    for _ in range(1, n_levels):
        expected.append(_expected_downsampled(expected[-1], is_mask=False,
                                              target_dtype=np.uint16))

    for L, exp in enumerate(expected):
        actual = np.asarray(root[str(L)][:])
        np.testing.assert_array_equal(actual, exp, err_msg=f"level {L} mismatch")


def test_ome_zarr_mask_uses_nearest(tmp_path):
    out = tmp_path / "mask.ome.zarr"
    rng = np.random.default_rng(7)
    data = rng.integers(0, 50, (2, 512, 512), dtype=np.uint16)
    writer = PyramidWriter.from_array(data, channel_names=["a", "b"], is_mask=True)
    writer.export_ome_zarr(out, chunk_size=128, is_mask=True)

    root = _open_zarr_group(out)
    assert root.attrs["multiscales"][0]["type"] == "nearest"

    n_levels = len(root.attrs["multiscales"][0]["datasets"])
    expected = [data]
    for _ in range(1, n_levels):
        expected.append(_expected_downsampled(expected[-1], is_mask=True,
                                              target_dtype=np.uint16))
    for L, exp in enumerate(expected):
        actual = np.asarray(root[str(L)][:])
        np.testing.assert_array_equal(actual, exp, err_msg=f"level {L} mask mismatch")


def test_ome_zarr_and_ometiff_byte_identical(tmp_path):
    """Both formats hold byte-identical pyramid data (different containers, same numbers)."""
    rng = np.random.default_rng(123)
    data = rng.integers(0, 1000, (3, 512, 512), dtype=np.uint16)

    writer_a = PyramidWriter.from_array(data, channel_names=["A", "B", "C"])
    writer_a.export_ometiff_pyramid(tmp_path / "out.ome.tiff", tile_size=128)

    writer_b = PyramidWriter.from_array(data, channel_names=["A", "B", "C"])
    writer_b.export_ome_zarr(tmp_path / "out.ome.zarr", chunk_size=128)

    root = _open_zarr_group(tmp_path / "out.ome.zarr")
    n_levels = len(root.attrs["multiscales"][0]["datasets"])
    for L in range(n_levels):
        tiff_level = tifffile.imread(str(tmp_path / "out.ome.tiff"), level=L)
        zarr_level = np.asarray(root[str(L)][:])
        np.testing.assert_array_equal(
            tiff_level, zarr_level, err_msg=f"level {L} differs between formats"
        )
