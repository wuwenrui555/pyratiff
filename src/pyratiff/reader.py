"""TiffZarrReader — lazy zarr-backed reader for TIFF files."""

import json
import pathlib
import re
from typing import Union
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import tifffile
import zarr


class TiffZarrReader:
    """Lazy TIFF reader backed by zarr, with channel-name indexing.

    Supports OME-TIFF and QPTIFF formats via the :meth:`from_ometiff` and
    :meth:`from_qptiff` class methods. Direct construction accepts any TIFF
    with manually supplied channel names.

    Attributes
    ----------
    zimg : zarr.Array
        Full image array, shape ``(C, H, W)`` or ``(H, W)``.
    channel_names : list[str]
        Channel names in the same order as the first axis of ``zimg``.
    zimg_dict : dict[str, zarr.Array]
        Per-channel lazy arrays keyed by channel name, each shape ``(H, W)``.

    Examples
    --------
    >>> reader = TiffZarrReader.from_ometiff("image.ome.tiff")
    >>> print(reader.channel_names)
    ['DAPI', 'CD45', 'PanCK']
    >>> patch = reader.zimg[reader.channel_index(["DAPI", "CD45"]), :512, :512]
    """

    def __init__(
        self,
        tiff_f: Union[str, pathlib.Path],
        channel_names: list[str] = None,
    ):
        """
        Parameters
        ----------
        tiff_f : str or pathlib.Path
            Path to the TIFF file.
        channel_names : list[str], optional
            Names of the channels. Defaults to ``["channel_0", ...]`` when
            ``None``.

        Raises
        ------
        FileNotFoundError
            When ``tiff_f`` does not exist.
        ValueError
            When the image has unsupported dimensions, or when the length of
            ``channel_names`` does not match the number of channels.
        """
        tiff_f = pathlib.Path(tiff_f)
        if not tiff_f.exists():
            raise FileNotFoundError(f"File not found: {tiff_f}")

        self.zimg = zarr.open(tifffile.imread(tiff_f, level=0, aszarr=True), mode="r")

        if self.zimg.ndim == 3:
            if channel_names is None:
                channel_names = [f"channel_{i}" for i in range(self.zimg.shape[0])]
            n_channel = self.zimg.shape[0]
        elif self.zimg.ndim == 2:
            if channel_names is None:
                channel_names = ["channel_0"]
            n_channel = 1
        else:
            raise ValueError(f"Unsupported number of dimensions: {self.zimg.ndim}")

        if len(channel_names) != n_channel:
            raise ValueError(
                f"channel_names: Expected {n_channel} channel names, got {len(channel_names)}"
            )
        self.channel_names = channel_names

        self.zimg_dict = {
            channel_name: zarr.open(
                tifffile.imread(tiff_f, key=i, level=0, aszarr=True), mode="r"
            )
            for i, channel_name in enumerate(self.channel_names)
        }

    @classmethod
    def from_ometiff(
        cls,
        tiff_f: Union[str, pathlib.Path],
        markerlist_f: Union[str, pathlib.Path] = None,
    ) -> "TiffZarrReader":
        """Initialize a :class:`TiffZarrReader` from an OME-TIFF file.

        Channel names are read from the OME-XML metadata embedded in the file,
        or from a plain-text marker list when ``markerlist_f`` is provided.

        Parameters
        ----------
        tiff_f : str or pathlib.Path
            Path to the ``.ome.tiff`` file.
        markerlist_f : str or pathlib.Path, optional
            Path to a plain-text file with one channel name per line.
            When provided, overrides the names extracted from OME-XML.

        Returns
        -------
        TiffZarrReader
        """
        if markerlist_f is None:
            channel_names = cls.extract_channel_names_ometiff(tiff_f)
        else:
            with open(markerlist_f) as f:
                channel_names = [x.strip() for x in f.readlines()]
        return cls(tiff_f, channel_names)

    @classmethod
    def from_qptiff(
        cls,
        tiff_f: Union[str, pathlib.Path],
        markerlist_f: Union[str, pathlib.Path] = None,
    ) -> "TiffZarrReader":
        """Initialize a :class:`TiffZarrReader` from a QPTIFF file.

        Channel names are parsed from the ``ScanProfile`` JSON embedded in
        the ``ImageDescription`` tag, or from a plain-text marker list when
        ``markerlist_f`` is provided.

        Parameters
        ----------
        tiff_f : str or pathlib.Path
            Path to the ``.qptiff`` file.
        markerlist_f : str or pathlib.Path, optional
            Path to a plain-text file with one channel name per line.
            When provided, overrides the names parsed from the file.

        Returns
        -------
        TiffZarrReader
        """
        if markerlist_f is None:
            channel_names = cls.extract_channel_names_qptiff(tiff_f)
        else:
            with open(markerlist_f) as f:
                channel_names = [x.strip() for x in f.readlines()]
        return cls(tiff_f, channel_names)

    @staticmethod
    def extract_channel_names_ometiff(path: Union[str, pathlib.Path]) -> list[str]:
        """Extract channel names from an OME-TIFF file.

        Parses the OME-XML metadata embedded in the file to retrieve channel
        names from ``<Channel Name="...">`` elements.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ``.ome.tiff`` file.

        Returns
        -------
        list[str]
            Channel names in acquisition order. Falls back to
            ``["Channel 0", "Channel 1", ...]`` when no ``Name`` attribute
            is present.
        """
        with tifffile.TiffFile(path) as tif:
            ome_metadata = ElementTree.fromstring(tif.ome_metadata)
            ome_channels = ome_metadata.findall(".//{*}Channel")
            metadata = pd.DataFrame([channel.attrib for channel in ome_channels])
            if "Name" in metadata.columns:
                return metadata["Name"].tolist()
            return [f"Channel {i}" for i in range(len(metadata))]

    @staticmethod
    def extract_channel_names_qptiff(path: Union[str, pathlib.Path]) -> list[str]:
        """Extract channel names from a QPTIFF file.

        Reads the ``ScanProfile`` JSON from the ``ImageDescription`` TIFF tag
        and filters out blank/background entries using ``markerName`` and
        ``id`` fields.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the ``.qptiff`` file.

        Returns
        -------
        list[str]
            Marker channel names in acquisition order.

        Raises
        ------
        ValueError
            When no ``ScanProfile`` element is found in the ``ImageDescription``.
        """
        with tifffile.TiffFile(path) as tif:
            xml_string = tif.series[0].pages[0].tags["ImageDescription"].value
            scan_profile = ElementTree.fromstring(xml_string).find(".//ScanProfile")
            if scan_profile is None:
                raise ValueError(
                    "ScanProfile element not found in the ImageDescription tag."
                )
            data = json.loads(scan_profile.text)
            wells = data["experimentDescription"]["wells"]
            qptiff_metadata = pd.concat(
                [
                    pd.DataFrame(well["items"]).assign(wellName=well["wellName"])
                    for well in wells
                ],
                ignore_index=True,
            )
            is_marker = (qptiff_metadata["markerName"] != "--") & (
                qptiff_metadata["id"].apply(
                    lambda x: re.search(r"^0+(-0+)+$", x.strip()) is None
                )
            )
            return (
                qptiff_metadata.loc[is_marker]
                .drop_duplicates(["id", "markerName"])["markerName"]
                .tolist()
            )

    def channel_index(self, channels: Union[str, list[str]]) -> Union[int, list[int]]:
        """Return the index (or indices) of one or more channels by name.

        Parameters
        ----------
        channels : str or list[str]
            A single channel name or a list of channel names.

        Returns
        -------
        int or list[int]
            Index of the channel in ``self.channel_names``, or a list of
            indices when ``channels`` is a list.

        Raises
        ------
        ValueError
            When ``channel_names`` is ``None`` or ``channels`` has an
            unsupported type.

        Examples
        --------
        >>> reader.channel_index("DAPI")
        0
        >>> reader.channel_index(["DAPI", "CD45"])
        [0, 2]
        """
        if self.channel_names is None:
            raise ValueError("Channel names not set.")
        if isinstance(channels, str):
            return self.channel_names.index(channels)
        if isinstance(channels, list):
            return [self.channel_names.index(ch) for ch in channels]
        raise ValueError(f"Expected str or list[str], got {type(channels)}")

    def slice_array(self, ymin: int, ymax: int, xmin: int, xmax: int) -> np.ndarray:
        """Crop a rectangular region from the full image array.

        Parameters
        ----------
        ymin, ymax : int
            Row (Y) bounds of the crop, in pixels.
        xmin, xmax : int
            Column (X) bounds of the crop, in pixels.

        Returns
        -------
        numpy.ndarray
            Cropped array of shape ``(C, ymax-ymin, xmax-xmin)`` or
            ``(ymax-ymin, xmax-xmin)`` for 2-D images.
        """
        return np.asarray(self.zimg[..., ymin:ymax, xmin:xmax])

    def slice_dict(
        self, ymin: int, ymax: int, xmin: int, xmax: int
    ) -> dict[str, np.ndarray]:
        """Crop a rectangular region from each per-channel array.

        Parameters
        ----------
        ymin, ymax : int
            Row (Y) bounds of the crop, in pixels.
        xmin, xmax : int
            Column (X) bounds of the crop, in pixels.

        Returns
        -------
        dict[str, numpy.ndarray]
            Mapping from channel name to cropped 2-D array of shape
            ``(ymax-ymin, xmax-xmin)``.
        """
        return {
            name: np.asarray(self.zimg_dict[name][ymin:ymax, xmin:xmax])
            for name in self.channel_names
        }
