"""Tests for TiffZarrReader."""

import pathlib

import numpy as np
import pytest

from pyratiff import TiffZarrReader


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        TiffZarrReader(tmp_path / "nonexistent.tiff")


def test_init_3d(ome_tiff_3ch):
    reader = TiffZarrReader(ome_tiff_3ch)
    assert reader.zimg.ndim == 3
    assert reader.zimg.shape == (3, 64, 64)
    assert reader.channel_names == ["channel_0", "channel_1", "channel_2"]
    assert list(reader.zimg_dict.keys()) == ["channel_0", "channel_1", "channel_2"]


def test_init_2d(ome_tiff_2d):
    reader = TiffZarrReader(ome_tiff_2d)
    assert reader.zimg.ndim == 2
    assert reader.zimg.shape == (64, 64)
    assert reader.channel_names == ["channel_0"]


def test_init_custom_channel_names(ome_tiff_3ch):
    names = ["A", "B", "C"]
    reader = TiffZarrReader(ome_tiff_3ch, channel_names=names)
    assert reader.channel_names == names


def test_init_channel_names_length_mismatch(ome_tiff_3ch):
    with pytest.raises(ValueError, match="channel_names"):
        TiffZarrReader(ome_tiff_3ch, channel_names=["only_one"])


# ---------------------------------------------------------------------------
# from_ometiff
# ---------------------------------------------------------------------------


def test_from_ometiff_reads_channel_names(ome_tiff_3ch):
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch)
    assert reader.channel_names == ["DAPI", "CD45", "PanCK"]


def test_from_ometiff_markerlist_overrides(ome_tiff_3ch, markerlist_f):
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch, markerlist_f=markerlist_f)
    assert reader.channel_names == ["DAPI", "CD45", "PanCK"]


def test_from_ometiff_markerlist_custom(ome_tiff_3ch, tmp_path):
    custom = tmp_path / "custom.txt"
    custom.write_text("X\nY\nZ\n")
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch, markerlist_f=custom)
    assert reader.channel_names == ["X", "Y", "Z"]


# ---------------------------------------------------------------------------
# from_qptiff
# ---------------------------------------------------------------------------


def test_from_qptiff_reads_channel_names(qptiff):
    reader = TiffZarrReader.from_qptiff(qptiff)
    # conftest creates ScanProfile with DAPI and CD45; blank/sentinel filtered out
    assert reader.channel_names == ["DAPI", "CD45"]


def test_from_qptiff_markerlist_overrides(qptiff, tmp_path):
    override = tmp_path / "override.txt"
    override.write_text("MarkerA\nMarkerB\n")
    reader = TiffZarrReader.from_qptiff(qptiff, markerlist_f=override)
    assert reader.channel_names == ["MarkerA", "MarkerB"]


def test_from_qptiff_missing_scan_profile(tmp_path):
    """A TIFF whose ImageDescription has no ScanProfile should raise ValueError."""
    import tifffile

    path = tmp_path / "bad.qptiff"
    data = np.zeros((2, 64, 64), dtype=np.uint16)
    tifffile.imwrite(str(path), data, description="<root><Other/></root>")
    with pytest.raises(ValueError, match="ScanProfile"):
        TiffZarrReader.extract_channel_names_qptiff(path)


# ---------------------------------------------------------------------------
# channel_index
# ---------------------------------------------------------------------------


def test_channel_index_single(ome_tiff_3ch):
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch)
    assert reader.channel_index("DAPI") == 0
    assert reader.channel_index("CD45") == 1
    assert reader.channel_index("PanCK") == 2


def test_channel_index_list(ome_tiff_3ch):
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch)
    assert reader.channel_index(["PanCK", "DAPI"]) == [2, 0]


def test_channel_index_invalid_type(ome_tiff_3ch):
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch)
    with pytest.raises(ValueError):
        reader.channel_index(0)


def test_channel_index_unknown_name(ome_tiff_3ch):
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch)
    with pytest.raises(ValueError):
        reader.channel_index("DoesNotExist")


# ---------------------------------------------------------------------------
# slice_array / slice_dict
# ---------------------------------------------------------------------------


def test_slice_array_shape(ome_tiff_3ch):
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch)
    result = reader.slice_array(ymin=4, ymax=14, xmin=8, xmax=28)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 10, 20)


def test_slice_dict_shape(ome_tiff_3ch):
    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch)
    result = reader.slice_dict(ymin=4, ymax=14, xmin=8, xmax=28)
    assert list(result.keys()) == ["DAPI", "CD45", "PanCK"]
    for arr in result.values():
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (10, 20)


def test_slice_array_values(ome_tiff_3ch):
    """Sliced values must match what tifffile returns directly."""
    import tifffile

    reader = TiffZarrReader.from_ometiff(ome_tiff_3ch)
    result = reader.slice_array(0, 10, 0, 10)
    expected = tifffile.imread(str(ome_tiff_3ch), level=0)[:, :10, :10]
    np.testing.assert_array_equal(result, expected)
