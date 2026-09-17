"""Read a PyTorch `pytorch_model.bin` (zip + pickle) into numpy arrays, without torch.

The archive holds `data.pkl`, which pickles an OrderedDict of tensors whose storages are
persistent-id references to raw little-endian buffers in `data/<key>`. We stub the torch
classes the pickle names and rebuild each tensor as a numpy array.
"""

from __future__ import annotations

import pickle
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_STORAGE_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "BFloat16Storage": np.uint16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _Storage:
    def __init__(self, zf: zipfile.ZipFile, name: str, dtype: type) -> None:
        self.zf, self.name, self.dtype = zf, name, dtype
        self._data: NDArray[Any] | None = None

    def data(self) -> NDArray[Any]:
        if self._data is None:
            with self.zf.open(self.name) as f:
                self._data = np.frombuffer(f.read(), dtype=self.dtype)
        return self._data


def _rebuild_tensor(
    storage: _Storage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    *_: Any,
) -> NDArray[Any]:
    buf = storage.data()
    itemsize = buf.dtype.itemsize
    view = np.lib.stride_tricks.as_strided(
        buf[storage_offset:], shape=tuple(size), strides=tuple(s * itemsize for s in stride)
    )
    return np.ascontiguousarray(view)


class _Unpickler(pickle.Unpickler):
    def __init__(self, file: Any, zf: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(file, encoding="utf-8")
        self.zf, self.prefix = zf, prefix

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, *_: data
        if module == "torch" and name in _STORAGE_DTYPES:
            return name
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        raise pickle.UnpicklingError(f"refusing to load {module}.{name}")

    def persistent_load(self, pid: Any) -> _Storage:
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage":
            raise pickle.UnpicklingError(f"unknown persistent id {kind}")
        return _Storage(self.zf, f"{self.prefix}/data/{key}", _STORAGE_DTYPES[storage_type])


def load_torch_checkpoint(path: Path) -> dict[str, NDArray[Any]]:
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
        prefix = pkl[: -len("/data.pkl")]
        with zf.open(pkl) as f:
            state: dict[str, NDArray[Any]] = _Unpickler(f, zf, prefix).load()
        return {k: np.asarray(v) for k, v in state.items()}
