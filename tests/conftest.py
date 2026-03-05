"""Shared fixtures for pyratiff tests."""

import json
import pathlib

import numpy as np
import pytest
import tifffile


def _make_scan_profile(channel_names: list[str]) -> str:
    """Build a minimal ScanProfile JSON string embedded in OME-like XML."""
    items = [{"markerName": name, "id": f"id-{i}"} for i, name in enumerate(channel_names)]
    # Add a blank/background item that should be filtered out
    items.append({"markerName": "--", "id": "id-blank"})
    items.append({"markerName": "real", "id": "00-00"})  # zero-sentinel id, filtered out

    data = {
        "experimentDescription": {
            "wells": [{"wellName": "A1", "items": items}]
        }
    }
    scan_profile_json = json.dumps(data)
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        "<root>"
        f"<ScanProfile>{scan_profile_json}</ScanProfile>"
        "</root>"
    )
    return xml


@pytest.fixture()
def ome_tiff_3ch(tmp_path: pathlib.Path) -> pathlib.Path:
    """3-channel (3, 64, 64) uint16 OME-TIFF with names DAPI/CD45/PanCK."""
    path = tmp_path / "test_3ch.ome.tiff"
    data = np.arange(3 * 64 * 64, dtype=np.uint16).reshape(3, 64, 64)
    metadata = {"Channel": {"Name": ["DAPI", "CD45", "PanCK"]}}
    tifffile.imwrite(str(path), data, metadata=metadata, photometric="minisblack")
    return path


@pytest.fixture()
def ome_tiff_2d(tmp_path: pathlib.Path) -> pathlib.Path:
    """Single-channel (64, 64) uint8 TIFF."""
    path = tmp_path / "test_2d.tiff"
    data = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    tifffile.imwrite(str(path), data)
    return path


@pytest.fixture()
def qptiff(tmp_path: pathlib.Path) -> pathlib.Path:
    """Synthetic QPTIFF: 2-channel (2, 64, 64) uint16 with ScanProfile JSON."""
    path = tmp_path / "test.qptiff"
    data = np.zeros((2, 64, 64), dtype=np.uint16)
    description = _make_scan_profile(["DAPI", "CD45"])
    tifffile.imwrite(
        str(path),
        data,
        description=description,
        photometric="minisblack",
    )
    return path


@pytest.fixture()
def markerlist_f(tmp_path: pathlib.Path) -> pathlib.Path:
    """Plain-text marker list: DAPI / CD45 / PanCK."""
    path = tmp_path / "markers.txt"
    path.write_text("DAPI\nCD45\nPanCK\n")
    return path
