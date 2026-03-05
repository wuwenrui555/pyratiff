"""PyramidWriter — assemble multi-channel pyramidal OME-TIFF files."""

import concurrent.futures
import itertools
import multiprocessing
import os
import pathlib
import uuid
from typing import Union

import numpy as np
import skimage.transform
import tifffile
import zarr
from tqdm import tqdm

_TQDM_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt}"
    " [{elapsed}<{remaining}, {rate_fmt}]"
)


class PyramidWriter:
    """Assemble a multi-channel pyramidal OME-TIFF from arrays or files.

    Construct via one of the class methods, then call
    :meth:`export_ometiff_pyramid` to write the output file.

    Parameters
    ----------
    in_imgs : list[numpy.ndarray]
        One 2-D ``(H, W)`` array per channel, all with the same shape and
        dtype.
    in_chns : list[str]
        Channel names, same length as ``in_imgs``.
    target_shape : tuple[int, int]
        ``(H, W)`` of each channel image.
    target_dtype : numpy.dtype
        Common dtype for all channels (typically the maximum across inputs).

    Examples
    --------
    >>> import numpy as np
    >>> from pyratiff import PyramidWriter
    >>> data = np.zeros((3, 4096, 4096), dtype=np.uint16)
    >>> writer = PyramidWriter.from_array(data, channel_names=["DAPI", "CD45", "PanCK"])
    >>> writer.export_ometiff_pyramid("out.ome.tiff", pixel_size=0.5)
    """

    def __init__(
        self,
        in_imgs: list[np.ndarray],
        in_chns: list[str],
        target_shape: tuple[int, int],
        target_dtype: np.dtype,
    ):
        self.in_imgs = [np.asarray(img).astype(target_dtype) for img in in_imgs]
        self.in_chns = in_chns
        self.target_shape = target_shape
        self.target_dtype = target_dtype

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_fs(
        cls,
        input_data: list[Union[str, pathlib.Path]],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """Build a :class:`PyramidWriter` from a list of single-channel TIFF files.

        Each file may be 2-D ``(H, W)`` or 3-D ``(C, H, W)``. For 3-D files
        every slice is treated as a separate channel named
        ``<channel_name>_0``, ``<channel_name>_1``, …

        Parameters
        ----------
        input_data : list[str or pathlib.Path]
            Paths to input TIFF files.
        channel_names : list[str], optional
            One name per file. Defaults to ``["channel_0", ...]``.
        is_mask : bool, optional
            Allow 32-bit dtypes and use nearest-neighbour downsampling.
            Default ``False``.

        Returns
        -------
        PyramidWriter

        Raises
        ------
        ValueError
            When ``channel_names`` length does not match ``input_data``, an
            image has an unsupported shape, or a non-mask image has a 32-bit
            dtype.
        """
        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(len(input_data))]
        if len(channel_names) != len(input_data):
            raise ValueError(
                f"channel_names: Expected {len(input_data)} names, got {len(channel_names)}"
            )

        in_imgs: list[np.ndarray] = []
        in_chns: list[str] = []
        target_shape: tuple[int, int] | None = None

        for i, path in tqdm(
            enumerate(input_data),
            total=len(input_data),
            desc="Loading images",
            bar_format=_TQDM_FORMAT,
        ):
            name = channel_names[i]
            zimg = zarr.open(tifffile.imread(path, level=0, aszarr=True), mode="r")
            if i == 0:
                target_shape = zimg.shape[-2:]

            if zimg.ndim == 2:
                img = np.asarray(zimg)
                cls._validate_image_2d(img.shape, img.dtype, target_shape, is_mask, name)
                in_imgs.append(img)
                in_chns.append(name)
            elif zimg.ndim == 3:
                for j in range(zimg.shape[0]):
                    img = np.asarray(zimg[j])
                    cls._validate_image_2d(
                        img.shape, img.dtype, target_shape, is_mask, f"{name}_{j}"
                    )
                    in_imgs.append(img)
                    in_chns.append(f"{name}_{j}")
            else:
                raise ValueError(f"{path}: Unsupported ndim={zimg.ndim}")

        target_dtype = max(img.dtype for img in in_imgs)
        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_array(
        cls,
        input_data: Union[np.ndarray, zarr.Array],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """Build a :class:`PyramidWriter` from a 2-D or 3-D array.

        Parameters
        ----------
        input_data : numpy.ndarray or zarr.Array
            Shape ``(H, W)`` (treated as a single channel) or ``(C, H, W)``.
        channel_names : list[str], optional
            One name per channel. Defaults to ``["channel_0", ...]``.
        is_mask : bool, optional
            Allow 32-bit dtypes and use nearest-neighbour downsampling.
            Default ``False``.

        Returns
        -------
        PyramidWriter

        Raises
        ------
        ValueError
            When the array is not 2-D or 3-D, ``channel_names`` length
            mismatches, or a non-mask image has an unsupported dtype.
        """
        if input_data.ndim == 2:
            input_data = np.asarray(input_data)[np.newaxis, ...]
        elif input_data.ndim == 3:
            input_data = np.asarray(input_data)
        else:
            raise ValueError(
                f"input_data: Expected 2-D or 3-D array, got shape {input_data.shape}"
            )

        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(input_data.shape[0])]
        if len(channel_names) != input_data.shape[0]:
            raise ValueError(
                f"channel_names: Expected {input_data.shape[0]} names, got {len(channel_names)}"
            )

        target_shape = input_data.shape[-2:]
        in_imgs: list[np.ndarray] = []
        in_chns: list[str] = []
        for name, img in zip(channel_names, input_data):
            cls._validate_image_2d(img.shape, img.dtype, target_shape, is_mask, name)
            in_imgs.append(img)
            in_chns.append(name)

        target_dtype = max(img.dtype for img in in_imgs)
        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_dict(
        cls,
        input_data: dict[str, Union[np.ndarray, zarr.Array]],
        channel_names: list[str] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """Build a :class:`PyramidWriter` from a dict of arrays.

        Parameters
        ----------
        input_data : dict[str, numpy.ndarray or zarr.Array]
            Keys are channel names; values are 2-D ``(H, W)`` or 3-D
            ``(C, H, W)`` arrays. 3-D values are split into separate
            channels named ``<key>_0``, ``<key>_1``, …
        channel_names : list[str], optional
            Override dict keys for channel ordering. Defaults to
            ``list(input_data.keys())``.
        is_mask : bool, optional
            Allow 32-bit dtypes and use nearest-neighbour downsampling.
            Default ``False``.

        Returns
        -------
        PyramidWriter

        Raises
        ------
        ValueError
            On shape or dtype validation failures.
        """
        if channel_names is None:
            channel_names = list(input_data.keys())
        if len(channel_names) != len(input_data):
            raise ValueError(
                f"channel_names: Expected {len(input_data)} names, got {len(channel_names)}"
            )

        in_imgs: list[np.ndarray] = []
        in_chns: list[str] = []
        target_shape: tuple[int, int] | None = None
        for name, img_in in zip(channel_names, input_data.values()):
            if not isinstance(img_in, (np.ndarray, zarr.Array)):
                raise ValueError(f"{name}: Unsupported type {type(img_in)}")
            img_in = np.asarray(img_in)
            if target_shape is None:
                target_shape = img_in.shape[-2:]
            if img_in.ndim == 2:
                cls._validate_image_2d(img_in.shape, img_in.dtype, target_shape, is_mask, name)
                in_imgs.append(img_in)
                in_chns.append(name)
            elif img_in.ndim == 3:
                for i in range(img_in.shape[0]):
                    img = img_in[i]
                    cls._validate_image_2d(
                        img.shape, img.dtype, target_shape, is_mask, f"{name}_{i}"
                    )
                    in_imgs.append(img)
                    in_chns.append(f"{name}_{i}")
            else:
                raise ValueError(f"{name}: Unsupported ndim={img_in.ndim}")

        target_dtype = max(img.dtype for img in in_imgs)
        return cls(in_imgs, in_chns, target_shape, target_dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_image_2d(
        shape: tuple[int, int],
        dtype: np.dtype,
        target_shape: tuple[int, int],
        is_mask: bool,
        msg_tag: str = "",
    ) -> None:
        """Raise ``ValueError`` when *shape* or *dtype* fails validation.

        Parameters
        ----------
        shape : tuple[int, int]
            ``(H, W)`` of the image to validate.
        dtype : numpy.dtype
            Data type of the image.
        target_shape : tuple[int, int]
            Expected ``(H, W)``.
        is_mask : bool
            When ``True``, ``uint32`` and ``int32`` are allowed.
        msg_tag : str
            Prefix for error messages.

        Raises
        ------
        ValueError
            On dtype or shape mismatch.
        """
        prefix = f"{msg_tag}: " if msg_tag else ""
        if dtype in (np.dtype("uint32"), np.dtype("int32")):
            if not is_mask:
                raise ValueError(
                    f"{prefix}32-bit dtype is only supported with is_mask=True"
                )
        elif dtype not in (
            np.dtype("uint8"),
            np.dtype("uint16"),
            np.dtype("float32"),
            np.dtype("float64"),
        ):
            raise ValueError(f"{prefix}Unsupported dtype: {dtype}")
        if shape != target_shape:
            raise ValueError(
                f"{prefix}Shape mismatch: expected {target_shape}, got {shape}"
            )

    @staticmethod
    def _build_ome_metadata(pixel_size: float | None, channel_names: list[str]) -> dict:
        """Return a tifffile-compatible OME metadata dict.

        Parameters
        ----------
        pixel_size : float or None
            Isotropic pixel size in microns. Omitted from metadata when ``None``.
        channel_names : list[str]
            Channel names written to the ``<Channel Name="...">`` OME-XML elements.

        Returns
        -------
        dict
            Dict accepted by the ``metadata`` argument of
            :class:`tifffile.TiffWriter`.
        """
        meta: dict = {"UUID": uuid.uuid4().urn}
        if pixel_size is not None:
            meta.update(
                {
                    "PhysicalSizeX": pixel_size,
                    "PhysicalSizeXUnit": "µm",
                    "PhysicalSizeY": pixel_size,
                    "PhysicalSizeYUnit": "µm",
                }
            )
        if channel_names:
            meta["Channel"] = {"Name": channel_names}
        return meta

    def _make_tile_generators(
        self,
        output_f: pathlib.Path,
        tile_size: int,
        is_mask: bool,
        num_threads: int,
    ):
        """Create tile generator functions for the base level and pyramid levels.

        Returns
        -------
        tiles0 : callable
            Generator yielding tiles for pyramid level 0.
        tiles : callable
            Generator factory; ``tiles(level)`` yields tiles for the given level
            by reading the partially-written output file and downsampling 2×.
        num_levels : int
            Total number of pyramid levels including the base.
        shapes : numpy.ndarray
            ``(num_levels, 2)`` array of ``(H, W)`` at each level.
        cshapes : numpy.ndarray
            ``(num_levels, 2)`` array of tile counts ``(rows, cols)`` at each level.
        """
        H, W = self.target_shape
        num_levels = max(1, int(np.ceil(np.log2(max(H, W) / tile_size)) + 1))
        factors = 2 ** np.arange(num_levels)
        shapes = np.ceil(np.array([H, W]) / factors[:, None]).astype(int)
        cshapes = np.ceil(shapes / tile_size).astype(int)

        in_imgs = self.in_imgs
        target_dtype = self.target_dtype
        num_channels = len(self.in_chns)
        pool = concurrent.futures.ThreadPoolExecutor(num_threads)

        def tiles0():
            ts = tile_size
            ch, cw = cshapes[0]
            for img in in_imgs:
                for j in range(ch):
                    for i in range(cw):
                        yield img[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]

        def tiles(level: int):
            with tifffile.TiffFile(output_f, is_ome=False) as tiff_out:
                zimg = zarr.open(
                    tiff_out.series[0].aszarr(level=level - 1), mode="r"
                )
                ts = tile_size * 2

                def tile(coords):
                    c, j, i = coords
                    if zimg.ndim == 2:
                        t = zimg[ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                    else:
                        t = zimg[c, ts * j : ts * (j + 1), ts * i : ts * (i + 1)]
                    if is_mask:
                        return t[::2, ::2].astype(target_dtype)
                    downsampled = skimage.transform.downscale_local_mean(t, (2, 2))
                    if np.issubdtype(target_dtype, np.floating):
                        return downsampled.astype(target_dtype)
                    return np.round(downsampled).astype(target_dtype)

                ch, cw = cshapes[level]
                coords = itertools.product(range(num_channels), range(ch), range(cw))
                yield from pool.map(tile, coords)

        return tiles0, tiles, num_levels, shapes, cshapes

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_ometiff_pyramid(
        self,
        output_f: Union[str, pathlib.Path],
        pixel_size: float = None,
        tile_size: int = 256,
        is_mask: bool = False,
        num_threads: int = 8,
        overwrite: bool = True,
    ) -> None:
        """Write the image stack as a tiled pyramidal OME-TIFF.

        The pyramid is assembled following the approach described in
        `LabSysPharm/ome-tiff-pyramid-tools
        <https://github.com/labsyspharm/ome-tiff-pyramid-tools>`_.
        Each level is half the resolution of the previous one; the base level
        (level 0) is written first, and each subsequent level is generated by
        reading the tiles written so far from the output file.

        Parameters
        ----------
        output_f : str or pathlib.Path
            Destination ``.ome.tiff`` path.
        pixel_size : float, optional
            Isotropic pixel size in microns, written to the OME-XML
            ``PhysicalSizeX/Y`` attributes. Omitted when ``None``.
        tile_size : int, optional
            Tile width and height in pixels. Must be a multiple of 16.
            Default ``256``.
        is_mask : bool, optional
            When ``True``, nearest-neighbour downsampling is used instead of
            local-mean averaging, preserving integer label values.
            Default ``False``.
        num_threads : int, optional
            Worker threads for tile compression and downsampling.
            Pass ``0`` to auto-detect from CPU affinity.
            Default ``8``.
        overwrite : bool, optional
            Overwrite an existing file when ``True``; raise
            :exc:`FileExistsError` when ``False``. Default ``True``.

        Raises
        ------
        FileExistsError
            When ``output_f`` exists and ``overwrite=False``.
        """
        output_f = pathlib.Path(output_f)
        if output_f.exists():
            if overwrite:
                print(f"Overwriting existing file: {output_f}")
                output_f.unlink()
            else:
                raise FileExistsError(f"Output file already exists: {output_f}")

        if num_threads == 0:
            if hasattr(os, "sched_getaffinity"):
                num_threads = len(os.sched_getaffinity(0))
            else:
                num_threads = multiprocessing.cpu_count()
            print(f"Using {num_threads} worker threads")

        tifffile.TIFF.MAXWORKERS = num_threads
        tifffile.TIFF.MAXIOWORKERS = num_threads * 5

        metadata = self._build_ome_metadata(pixel_size, self.in_chns)
        num_channels = len(self.in_chns)
        tiles0, tiles, num_levels, shapes, cshapes = self._make_tile_generators(
            output_f=output_f,
            tile_size=tile_size,
            is_mask=is_mask,
            num_threads=num_threads,
        )

        total_tiles = sum(cs[0] * cs[1] for cs in cshapes)
        pbar = tqdm(total=total_tiles, desc="Writing tiles", bar_format=_TQDM_FORMAT)

        with tifffile.TiffWriter(output_f, ome=True, bigtiff=True) as writer:
            for level, shape in enumerate(shapes):
                if level == 0:
                    writer.write(
                        data=tiles0(),
                        shape=(num_channels,) + tuple(shape),
                        subifds=num_levels - 1,
                        dtype=self.target_dtype,
                        tile=(tile_size, tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                        metadata=metadata,
                    )
                else:
                    writer.write(
                        data=tiles(level),
                        shape=(num_channels,) + tuple(shape),
                        subfiletype=1,
                        dtype=self.target_dtype,
                        tile=(tile_size, tile_size),
                        compression="adobe_deflate",
                        predictor=True,
                    )
                pbar.update(cshapes[level][0] * cshapes[level][1])

        pbar.close()
