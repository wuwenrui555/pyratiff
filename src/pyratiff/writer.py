"""PyramidWriter — assemble multi-channel pyramidal OME-NGFF (zarr) and OME-TIFF files.

v2.0 architecture
-----------------
- **Input is lazy**: ``self.in_imgs`` is a list of array-likes that support
  numpy-style 2-D slicing. numpy arrays passed to ``from_array`` are wrapped
  in an in-memory zarr; OME-TIFF and OME-NGFF inputs stay file-backed via
  ``tifffile.imread(aszarr=True)`` and ``zarr.open_group``.
- **Pyramid is streamed**: each pyramid level is written to the output zarr
  one tile at a time. Memory peak is ``O(num_channels * tile_size² * dtype)``,
  not the full image. Larger-than-RAM inputs go through fine.
- **OME-Zarr is the canonical output**: ``export_ome_zarr`` writes a full
  OME-NGFF v0.4 group; ``export_ometiff_pyramid`` writes the same image stack
  by first staging to a temporary OME-Zarr and then converting to OME-TIFF.
"""

import logging
import multiprocessing
import os
import pathlib
import shutil
import tempfile
import uuid
from typing import Any, Optional, Union

import numpy as np
import skimage.transform
import tifffile
import zarr
from tqdm import tqdm

logger = logging.getLogger(__name__)

_TQDM_FORMAT = (
    "{desc}: {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt}"
    " [{elapsed}<{remaining}, {rate_fmt}]"
)

_VALID_IMAGE_DTYPES = {
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
    np.dtype("float32"),
    np.dtype("float64"),
}
_VALID_MASK_DTYPES = {
    np.dtype("uint8"),
    np.dtype("uint16"),
    np.dtype("uint32"),
}


class _ChannelView:
    """Lazy 2-D view of one channel of a 3-D ``(C, H, W)`` zarr-like array.

    Forwards numpy-style slicing to the underlying parent without ever
    materializing the full channel in memory.
    """

    __slots__ = ("_parent", "_c", "shape", "dtype", "ndim")

    def __init__(self, parent: Any, c: int):
        self._parent = parent
        self._c = c
        self.shape = tuple(parent.shape[1:])
        self.dtype = parent.dtype
        self.ndim = 2

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._parent[(self._c, *key)]
        return self._parent[self._c, key]


class PyramidWriter:
    """Assemble a multi-channel pyramidal OME-NGFF / OME-TIFF from arrays or files.

    Construct via one of the class methods, then call
    :meth:`export_ome_zarr` (canonical) or :meth:`export_ometiff_pyramid`
    (derived) to write the output.

    Attributes
    ----------
    in_imgs : list
        One 2-D array-like per channel, each lazily indexed by ``[y, x]``.
        May be numpy arrays (in-memory) or zarr arrays (file-backed).
    in_chns : list[str]
        Channel names, one per entry in ``in_imgs``.
    target_shape : tuple[int, int]
        ``(H, W)`` of each channel image.
    target_dtype : numpy.dtype
        Common dtype for all channels (output dtype after pyramid casting).

    Examples
    --------
    >>> import numpy as np
    >>> from pyratiff import PyramidWriter
    >>> data = np.zeros((3, 4096, 4096), dtype=np.uint16)
    >>> writer = PyramidWriter.from_array(data, channel_names=["DAPI", "CD45", "PanCK"])
    >>> writer.export_ome_zarr("out.ome.zarr", pixel_size=0.5)        # napari-friendly
    >>> writer.export_ometiff_pyramid("out.ome.tiff", pixel_size=0.5)  # QuPath-friendly
    """

    DEFAULT_TILE_SIZE: int = 256

    def __init__(
        self,
        in_imgs: list,
        in_chns: list[str],
        target_shape: tuple[int, int],
        target_dtype: np.dtype,
    ):
        self.in_imgs = in_imgs
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
        channel_names: Optional[list[str]] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """Build a writer from a list of single-channel TIFF files (lazy).

        Each file is opened via :func:`tifffile.imread` in ``aszarr=True``
        mode, so disk reads are deferred until pyramid generation. 3-D files
        are split into separate channels named ``<channel_name>_0``, ``_1`` …
        """
        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(len(input_data))]
        if len(channel_names) != len(input_data):
            raise ValueError(
                f"channel_names: Expected {len(input_data)} names, got {len(channel_names)}"
            )

        in_imgs: list = []
        in_chns: list[str] = []
        target_shape: Optional[tuple[int, int]] = None
        for i, path in enumerate(input_data):
            name = channel_names[i]
            zimg = zarr.open(tifffile.imread(path, level=0, aszarr=True), mode="r")
            if target_shape is None:
                target_shape = tuple(zimg.shape[-2:])

            if zimg.ndim == 2:
                cls._validate_image_2d(zimg.shape, zimg.dtype, target_shape, is_mask, name)
                in_imgs.append(zimg)
                in_chns.append(name)
            elif zimg.ndim == 3:
                for j in range(zimg.shape[0]):
                    sub = _ChannelView(zimg, j)
                    cls._validate_image_2d(sub.shape, sub.dtype, target_shape, is_mask,
                                           f"{name}_{j}")
                    in_imgs.append(sub)
                    in_chns.append(f"{name}_{j}")
            else:
                raise ValueError(f"{path}: Unsupported ndim={zimg.ndim}")

        target_dtype = max(img.dtype for img in in_imgs)
        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_array(
        cls,
        input_data: Union[np.ndarray, "zarr.Array"],
        channel_names: Optional[list[str]] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """Build a writer from a 2-D or 3-D array.

        numpy arrays are wrapped in an in-memory zarr array (so the rest of
        the pipeline can stay lazy-friendly). zarr arrays are kept as-is.
        """
        if input_data.ndim == 2:
            input_data = input_data[np.newaxis, ...]
        elif input_data.ndim != 3:
            raise ValueError(
                f"input_data: Expected 2-D or 3-D array, got shape {input_data.shape}"
            )

        if channel_names is None:
            channel_names = [f"channel_{i}" for i in range(input_data.shape[0])]
        if len(channel_names) != input_data.shape[0]:
            raise ValueError(
                f"channel_names: Expected {input_data.shape[0]} names, got {len(channel_names)}"
            )

        target_shape = tuple(input_data.shape[-2:])
        in_imgs: list = []
        in_chns: list[str] = []
        for c, name in enumerate(channel_names):
            ch = input_data[c]
            cls._validate_image_2d(tuple(ch.shape), ch.dtype, target_shape, is_mask, name)
            in_imgs.append(ch)  # numpy slice or zarr view — both support [y, x] slicing
            in_chns.append(name)

        target_dtype = max(np.asarray(img).dtype for img in in_imgs) \
            if isinstance(input_data, np.ndarray) else input_data.dtype
        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_dict(
        cls,
        input_data: dict[str, Union[np.ndarray, "zarr.Array"]],
        channel_names: Optional[list[str]] = None,
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """Build a writer from a dict of 2-D or 3-D arrays."""
        if channel_names is None:
            channel_names = list(input_data.keys())
        if len(channel_names) != len(input_data):
            raise ValueError(
                f"channel_names: Expected {len(input_data)} names, got {len(channel_names)}"
            )

        in_imgs: list = []
        in_chns: list[str] = []
        target_shape: Optional[tuple[int, int]] = None
        for name, img_in in zip(channel_names, input_data.values()):
            if not isinstance(img_in, (np.ndarray, zarr.Array)):
                raise ValueError(f"{name}: Unsupported type {type(img_in)}")
            if target_shape is None:
                target_shape = tuple(img_in.shape[-2:])
            if img_in.ndim == 2:
                cls._validate_image_2d(tuple(img_in.shape), img_in.dtype,
                                       target_shape, is_mask, name)
                in_imgs.append(img_in)
                in_chns.append(name)
            elif img_in.ndim == 3:
                for c in range(img_in.shape[0]):
                    sub = (img_in[c] if isinstance(img_in, np.ndarray)
                           else _ChannelView(img_in, c))
                    cls._validate_image_2d(tuple(sub.shape), sub.dtype,
                                           target_shape, is_mask, f"{name}_{c}")
                    in_imgs.append(sub)
                    in_chns.append(f"{name}_{c}")
            else:
                raise ValueError(f"{name}: Unsupported ndim={img_in.ndim}")

        target_dtype = max(img.dtype for img in in_imgs)
        return cls(in_imgs, in_chns, target_shape, target_dtype)

    @classmethod
    def from_ome_zarr(
        cls,
        path: Union[str, pathlib.Path],
        is_mask: bool = False,
    ) -> "PyramidWriter":
        """Build a writer from an existing OME-NGFF v0.4 zarr group (lazy).

        Uses level 0 as the source. Channel names come from
        ``omero.channels[].label`` when present, otherwise
        ``["channel_0", ...]``.
        """
        path = pathlib.Path(path)
        store = zarr.storage.LocalStore(str(path))
        root = zarr.open_group(store=store, mode="r")
        level0 = root["0"]  # zarr.Array of shape (C, H, W)
        if level0.ndim != 3:
            raise ValueError(
                f"OME-Zarr level 0 must be 3-D (C, H, W), got {level0.shape}"
            )

        omero = root.attrs.get("omero") or {}
        omero_channels = omero.get("channels") or []
        channel_names = [
            (omero_channels[i].get("label") if i < len(omero_channels) else None)
            or f"channel_{i}"
            for i in range(level0.shape[0])
        ]

        target_shape = tuple(level0.shape[1:])
        in_imgs = [_ChannelView(level0, c) for c in range(level0.shape[0])]
        for name, img in zip(channel_names, in_imgs):
            cls._validate_image_2d(img.shape, img.dtype, target_shape, is_mask, name)
        return cls(in_imgs, channel_names, target_shape, level0.dtype)

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
        prefix = f"{msg_tag}: " if msg_tag else ""
        if is_mask:
            if dtype not in _VALID_MASK_DTYPES:
                raise ValueError(
                    f"{prefix}Mask dtype must be unsigned integer "
                    f"(uint8/uint16/uint32), got {dtype}"
                )
        elif dtype not in _VALID_IMAGE_DTYPES:
            raise ValueError(f"{prefix}Unsupported dtype: {dtype}")
        if tuple(shape) != tuple(target_shape):
            raise ValueError(
                f"{prefix}Shape mismatch: expected {target_shape}, got {shape}"
            )

    @staticmethod
    def _build_ome_metadata(pixel_size: Optional[float], channel_names: list[str]) -> dict:
        meta: dict = {"UUID": uuid.uuid4().urn}
        if pixel_size is not None:
            meta.update({
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            })
        if channel_names:
            meta["Channel"] = {"Name": channel_names}
        return meta

    @staticmethod
    def _downsample_tile(
        tile: np.ndarray,
        is_mask: bool,
        target_dtype: np.dtype,
    ) -> np.ndarray:
        """2× downsample a single 2-D tile (mask: nearest, image: local mean)."""
        if is_mask:
            return tile[::2, ::2].astype(target_dtype)
        ds = skimage.transform.downscale_local_mean(tile, (2, 2))
        if np.issubdtype(target_dtype, np.floating):
            return ds.astype(target_dtype)
        return np.round(ds).astype(target_dtype)

    @staticmethod
    def _level_shape(base_shape: tuple[int, int], level: int) -> tuple[int, int]:
        H, W = base_shape
        return (int(np.ceil(H / 2 ** level)), int(np.ceil(W / 2 ** level)))

    @staticmethod
    def _num_levels(base_shape: tuple[int, int], tile_size: int) -> int:
        return max(1, int(np.ceil(np.log2(max(base_shape) / tile_size)) + 1))

    def _stream_pyramid_to_zarr_group(
        self,
        root: "zarr.Group",
        tile_size: int,
        is_mask: bool,
        num_levels: int,
        progress_label: str = "Writing tiles",
    ) -> None:
        """Build pyramid in ``root``, tile-by-tile, never holding a full level in RAM.

        Level 0 is streamed from ``self.in_imgs``; subsequent levels are
        streamed from the previously-written zarr level. Memory peak is
        ``O(num_channels * tile_size² * dtype_size)`` per active tile.
        """
        H, W = self.target_shape
        num_channels = len(self.in_chns)
        total_tiles = sum(
            int(np.ceil(self._level_shape((H, W), L)[0] / tile_size))
            * int(np.ceil(self._level_shape((H, W), L)[1] / tile_size))
            * num_channels
            for L in range(num_levels)
        )
        pbar = tqdm(total=total_tiles, desc=progress_label, bar_format=_TQDM_FORMAT)
        try:
            # Level 0: stream from self.in_imgs.
            level0 = root.create_array(
                name="0",
                shape=(num_channels, H, W),
                chunks=(num_channels, tile_size, tile_size),
                dtype=self.target_dtype,
            )
            for c in range(num_channels):
                src = self.in_imgs[c]
                for j in range(0, H, tile_size):
                    for i in range(0, W, tile_size):
                        j2 = min(j + tile_size, H)
                        i2 = min(i + tile_size, W)
                        tile = np.asarray(src[j:j2, i:i2]).astype(self.target_dtype)
                        level0[c, j:j2, i:i2] = tile
                        pbar.update(1)

            # Levels 1..N-1: stream from previously-written level.
            for L in range(1, num_levels):
                prev = root[str(L - 1)]
                H_L, W_L = self._level_shape((H, W), L)
                curr = root.create_array(
                    name=str(L),
                    shape=(num_channels, H_L, W_L),
                    chunks=(num_channels, tile_size, tile_size),
                    dtype=self.target_dtype,
                )
                for c in range(num_channels):
                    for j_out in range(0, H_L, tile_size):
                        for i_out in range(0, W_L, tile_size):
                            j_in = j_out * 2
                            i_in = i_out * 2
                            j_in_end = min(j_in + tile_size * 2, prev.shape[1])
                            i_in_end = min(i_in + tile_size * 2, prev.shape[2])
                            src_tile = np.asarray(prev[c, j_in:j_in_end, i_in:i_in_end])
                            ds = self._downsample_tile(src_tile, is_mask, self.target_dtype)
                            curr[c, j_out:j_out + ds.shape[0],
                                 i_out:i_out + ds.shape[1]] = ds
                            pbar.update(1)
        finally:
            pbar.close()

    def _write_ome_ngff_metadata(
        self,
        root: "zarr.Group",
        name: str,
        pixel_size: Optional[float],
        is_mask: bool,
        num_levels: int,
    ) -> None:
        """Attach OME-NGFF v0.4 ``multiscales`` and ``omero`` attrs to ``root``."""
        datasets = []
        for L in range(num_levels):
            scale_xy = float(2 ** L) * (pixel_size if pixel_size is not None else 1.0)
            datasets.append({
                "path": str(L),
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, scale_xy, scale_xy]}
                ],
            })

        spatial_axis: dict = {"type": "space"}
        if pixel_size is not None:
            spatial_axis["unit"] = "micrometer"
        axes = [
            {"name": "c", "type": "channel"},
            {**spatial_axis, "name": "y"},
            {**spatial_axis, "name": "x"},
        ]

        root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": name,
            "axes": axes,
            "datasets": datasets,
            "type": "nearest" if is_mask else "mean",
        }]
        root.attrs["omero"] = {
            "channels": [{"label": n} for n in self.in_chns],
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export_ome_zarr(
        self,
        output_dir: Union[str, pathlib.Path],
        pixel_size: Optional[float] = None,
        chunk_size: int = DEFAULT_TILE_SIZE,
        is_mask: bool = False,
        overwrite: bool = True,
    ) -> None:
        """Write the image stack as an OME-NGFF v0.4 zarr group (canonical output).

        Pyramid is generated tile-by-tile; memory peak is
        ``O(num_channels * chunk_size² * dtype)``.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Destination directory. Convention: end the name with
            ``.ome.zarr``.
        pixel_size : float, optional
            Isotropic pixel size in microns. When set, axes carry
            ``unit="micrometer"`` and the scale transformations encode it.
        chunk_size : int, optional
            Side length of zarr chunks in pixels. Also drives pyramid level
            count: ``ceil(log2(max(H, W) / chunk_size)) + 1``. Default ``256``.
        is_mask : bool, optional
            Nearest-neighbour subsampling between levels (preserves integer
            label values); ``multiscales[0].type`` is ``"nearest"``.
        overwrite : bool, optional
            Replace ``output_dir`` when it exists. Default ``True``.
        """
        output_dir = pathlib.Path(output_dir)
        if output_dir.exists():
            if overwrite:
                logger.info("Overwriting existing zarr: %s", output_dir)
                shutil.rmtree(output_dir)
            else:
                raise FileExistsError(f"Output directory already exists: {output_dir}")

        num_levels = self._num_levels(self.target_shape, chunk_size)

        store = zarr.storage.LocalStore(str(output_dir))
        root = zarr.open_group(store=store, mode="w", zarr_format=2)

        self._stream_pyramid_to_zarr_group(
            root=root,
            tile_size=chunk_size,
            is_mask=is_mask,
            num_levels=num_levels,
        )
        self._write_ome_ngff_metadata(
            root=root,
            name=output_dir.stem,
            pixel_size=pixel_size,
            is_mask=is_mask,
            num_levels=num_levels,
        )

    def export_ometiff_pyramid(
        self,
        output_f: Union[str, pathlib.Path],
        pixel_size: Optional[float] = None,
        tile_size: int = DEFAULT_TILE_SIZE,
        is_mask: bool = False,
        num_threads: int = 8,
        overwrite: bool = True,
    ) -> None:
        """Write the image stack as a tiled pyramidal OME-TIFF (derived output).

        Internally writes a temporary OME-Zarr (the canonical pyramid) and
        then converts it tile-by-tile to OME-TIFF. The temporary zarr is
        discarded after conversion. Memory profile matches
        :meth:`export_ome_zarr`.

        Parameters
        ----------
        output_f : str or pathlib.Path
            Destination ``.ome.tiff`` path.
        pixel_size : float, optional
            Isotropic pixel size in microns, written to OME-XML
            ``PhysicalSizeX/Y``.
        tile_size : int, optional
            Tile size in pixels; also drives pyramid level count.
            Default ``256``.
        is_mask : bool, optional
            Nearest-neighbour downsampling for label preservation.
        num_threads : int, optional
            Worker threads for tifffile compression. Pass ``0`` to auto-detect.
            Default ``8``.
        overwrite : bool, optional
            Overwrite an existing file when ``True``. Default ``True``.
        """
        output_f = pathlib.Path(output_f)
        if output_f.exists():
            if overwrite:
                logger.info("Overwriting existing file: %s", output_f)
                output_f.unlink()
            else:
                raise FileExistsError(f"Output file already exists: {output_f}")

        if num_threads == 0:
            if hasattr(os, "sched_getaffinity"):
                num_threads = len(os.sched_getaffinity(0))
            else:
                num_threads = multiprocessing.cpu_count()
            logger.info("Using %d worker threads", num_threads)

        prev_max_workers = tifffile.TIFF.MAXWORKERS
        prev_max_io_workers = tifffile.TIFF.MAXIOWORKERS
        tifffile.TIFF.MAXWORKERS = num_threads
        tifffile.TIFF.MAXIOWORKERS = num_threads * 5

        try:
            with tempfile.TemporaryDirectory(prefix="pyratiff_") as tmp:
                tmp_zarr = pathlib.Path(tmp) / "stage.ome.zarr"
                # Write canonical OME-Zarr first.
                self.export_ome_zarr(
                    output_dir=tmp_zarr,
                    pixel_size=pixel_size,
                    chunk_size=tile_size,
                    is_mask=is_mask,
                    overwrite=True,
                )
                # Then derive the OME-TIFF from it.
                self._zarr_group_to_ometiff(
                    zarr_dir=tmp_zarr,
                    output_f=output_f,
                    pixel_size=pixel_size,
                    tile_size=tile_size,
                )
        finally:
            tifffile.TIFF.MAXWORKERS = prev_max_workers
            tifffile.TIFF.MAXIOWORKERS = prev_max_io_workers

    def _zarr_group_to_ometiff(
        self,
        zarr_dir: pathlib.Path,
        output_f: pathlib.Path,
        pixel_size: Optional[float],
        tile_size: int,
    ) -> None:
        """Convert an OME-NGFF zarr group at ``zarr_dir`` to an OME-TIFF at ``output_f``."""
        store = zarr.storage.LocalStore(str(zarr_dir))
        root = zarr.open_group(store=store, mode="r")
        multiscales = root.attrs["multiscales"][0]
        n_levels = len(multiscales["datasets"])
        num_channels = len(self.in_chns)
        metadata = self._build_ome_metadata(pixel_size, self.in_chns)

        def tile_gen(level_arr):
            """Yield tiles in (channel, row, col) order from a (C, H_L, W_L) zarr array."""
            for c in range(level_arr.shape[0]):
                for j in range(0, level_arr.shape[1], tile_size):
                    for i in range(0, level_arr.shape[2], tile_size):
                        yield np.asarray(level_arr[
                            c,
                            j:j + tile_size,
                            i:i + tile_size,
                        ])

        with tifffile.TiffWriter(output_f, ome=True, bigtiff=True) as writer:
            for L in range(n_levels):
                arr = root[str(L)]
                kwargs = dict(
                    data=tile_gen(arr),
                    shape=tuple(arr.shape),
                    dtype=self.target_dtype,
                    tile=(tile_size, tile_size),
                    compression="adobe_deflate",
                    predictor=True,
                )
                if L == 0:
                    kwargs["subifds"] = n_levels - 1
                    kwargs["metadata"] = metadata
                else:
                    kwargs["subfiletype"] = 1
                writer.write(**kwargs)
                # Best-effort: free the zarr level we just consumed.
                # zarr 3 doesn't have an explicit close per array, but dropping
                # the reference lets the OS reclaim mmap pages.
                del arr
