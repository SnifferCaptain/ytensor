"""Minimal .yt (YTensor) reader/writer for numpy arrays.

Writes files compatible with the C++ `YTensorIO` format described in
`include/ytensor_io.hpp`.

Notes:
- No compression (compressMethod == "").
- Data layout is C-order (row-major).
- Magic is "YTENSORF", version 0, metadata is written empty.
"""

import struct
import numpy as np
from typing import Optional

_MAGIC = b"YTENSORF"  # 8 bytes
_VERSION = 0

# helpers for binary format used by ytensor_io.hpp

def _write_string(f, s: str):
    b = s.encode('utf-8')
    f.write(struct.pack('<I', len(b)))
    if len(b):
        f.write(b)


def _read_string(f) -> str:
    (length,) = struct.unpack('<I', f.read(4))
    if length == 0:
        return ""
    b = f.read(length)
    return b.decode('utf-8')


def _write_array_bytes(f, data_bytes: bytes):
    # array2data in C++ stores byte-count (uint64) followed by raw bytes
    f.write(struct.pack('<Q', len(data_bytes)))
    if len(data_bytes):
        f.write(data_bytes)


def _read_array_bytes(f) -> bytes:
    (count,) = struct.unpack('<Q', f.read(8))
    if count == 0:
        return b''
    return f.read(count)


class YTFile:
    """Manage multiple numpy arrays in a single .yt file.

    Usage:
      with YTFile('data.yt', mode='w') as ytf:
          ytf.add('a', arr1)
          ytf.add('b', arr2)

    Mode:
      'w' - create new file (overwrite)
      'a' - append to existing file (reads existing index and rewrites file on close)
      'r' - read-only (can list and load)
    """

    def __init__(self, path: str, mode: str = 'w'):
        assert mode in ('w', 'a', 'r')
        self.path = path
        self.mode = mode
        # each entry: dict with keys: name,type_name,type_size,tensor_type,shape,compress_method,compressed_size,data (optional),data_offset(optional)
        self._infos = []
        # if opening existing file, read headers and index
        if mode in ('a', 'r'):
            try:
                with open(self.path, 'rb') as f:
                    magic = f.read(8)
                    if magic != _MAGIC:
                        raise ValueError('Bad magic')
                    (version,) = struct.unpack('<B', f.read(1))
                    _ = _read_string(f)  # metadata
                    # read index
                    f.seek(-4, 2)
                    (tensor_count,) = struct.unpack('<I', f.read(4))
                    index_size = 4 + tensor_count * 8
                    f.seek(-index_size, 2)
                    offsets = [struct.unpack('<Q', f.read(8))[0] for _ in range(tensor_count)]
                    # read each tensor header to collect metadata
                    for off in offsets:
                        f.seek(off, 0)
                        name = _read_string(f)
                        type_name = _read_string(f)
                        (type_size,) = struct.unpack('<i', f.read(4))
                        tensor_type = _read_string(f)
                        shape_bytes = _read_array_bytes(f)
                        shape = []
                        if len(shape_bytes) > 0:
                            cnt = len(shape_bytes) // 4
                            shape = list(struct.unpack('<' + 'i' * cnt, shape_bytes))
                        compress_method = _read_string(f)
                        (compressed_size,) = struct.unpack('<Q', f.read(8))
                        data_offset = f.tell()
                        # skip data
                        f.seek(compressed_size, 1)
                        self._infos.append({'name': name,
                                            'type_name': type_name,
                                            'type_size': type_size,
                                            'tensor_type': tensor_type,
                                            'shape': shape,
                                            'compress_method': compress_method,
                                            'compressed_size': compressed_size,
                                            'data_offset': data_offset})
            except FileNotFoundError:
                # no existing file
                pass

    def add(self, name: str, array: np.ndarray):
        """Stage a numpy array to be written into the file on close."""
        dtype = array.dtype
        type_name = dtype.name
        type_size = dtype.itemsize
        tensor_type = 'dense'
        shape = list(int(s) for s in array.shape)
        data_bytes = array.tobytes(order='C')
        info = {'name': name,
                'type_name': type_name,
                'type_size': type_size,
                'tensor_type': tensor_type,
                'shape': shape,
                'compress_method': '',
                'compressed_size': len(data_bytes),
                'data': data_bytes}
        # if existing name, overwrite
        for i, e in enumerate(self._infos):
            if e['name'] == name:
                self._infos[i] = info
                return
        self._infos.append(info)

    def get_tensor_names(self):
        return [e['name'] for e in self._infos]

    def load(self, name: Optional[str] = None) -> np.ndarray:
        """Load a tensor by name (or first if name None)."""
        if not self._infos:
            raise RuntimeError('No tensors available')
        idx = 0
        if name is not None:
            for i, e in enumerate(self._infos):
                if e['name'] == name:
                    idx = i
                    break
            else:
                raise KeyError(f"Tensor named '{name}' not found")
        info = self._infos[idx]
        # if data already staged in memory
        if 'data' in info:
            data = info['data']
        else:
            # read from file on disk
            with open(self.path, 'rb') as f:
                f.seek(info['data_offset'], 0)
                data = f.read(info['compressed_size'])
        if info.get('compress_method', '') != '':
            raise NotImplementedError('Compression not supported by this converter')
        try:
            dt = np.dtype(info['type_name'])
            if dt.itemsize != info['type_size']:
                dt = np.dtype((dt.kind, info['type_size']))
        except Exception:
            dt = np.dtype(('V', info['type_size']))
        arr = np.frombuffer(data, dtype=dt)
        if info['shape']:
            arr = arr.reshape(tuple(info['shape']), order='C')
        return arr

    def save_all(self):
        """Write all staged (and existing) tensors into file (overwrites path)."""
        # We'll open original file when needing to copy existing raw bytes
        orig_path = self.path
        # Prepare buffers for any entries without in-memory 'data'
        for e in self._infos:
            if 'data' not in e:
                # read from original file
                with open(orig_path, 'rb') as f:
                    f.seek(e['data_offset'], 0)
                    e['data'] = f.read(e['compressed_size'])

        # Now write new file
        with open(self.path, 'wb') as f:
            # header
            f.write(_MAGIC)
            f.write(struct.pack('<B', _VERSION))
            _write_string(f, '')

            offsets = []
            for e in self._infos:
                offsets.append(f.tell())
                _write_string(f, e['name'])
                _write_string(f, e['type_name'])
                f.write(struct.pack('<i', e['type_size']))
                _write_string(f, e.get('tensor_type', 'dense'))
                # shape
                shape = e.get('shape', [])
                shape_bytes = struct.pack('<' + 'i' * len(shape), *shape) if shape else b''
                _write_array_bytes(f, shape_bytes)
                _write_string(f, e.get('compress_method', ''))
                data = e['data']
                f.write(struct.pack('<Q', len(data)))
                if len(data):
                    f.write(data)
            for off in offsets:
                f.write(struct.pack('<Q', off))
            f.write(struct.pack('<I', len(offsets)))

    def close(self):
        if self.mode in ('w', 'a'):
            self.save_all()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()



def save_yt(path: str, array: np.ndarray, name: str = '0') -> None:
    """
    Save a single numpy ndarray to a .yt file compatible with the C++ YTensorIO.
    - No compression will be used (compressMethod = "")
    - Data is stored in C-order (row-major)
    - Overwrites existing file
    """
    # prepare metadata
    dtype = array.dtype
    type_name = dtype.name  # e.g. 'float32', 'int64'
    type_size = dtype.itemsize
    tensor_type = 'dense'
    shape = tuple(int(s) for s in array.shape)

    with open(path, 'wb') as f:
        # header
        f.write(_MAGIC)
        f.write(struct.pack('<B', _VERSION))
        # metadata (empty string)
        _write_string(f, "")

        offsets = []

        # write one tensor
        offsets.append(f.tell())

        # name
        _write_string(f, name)
        # typeName
        _write_string(f, type_name)
        # typeSize (int32)
        f.write(struct.pack('<i', type_size))
        # tensorType
        _write_string(f, tensor_type)
        # shape as array[int32] -> C++ expects uint64 byte-count then raw bytes
        # pack int32 little-endian
        shape_bytes = struct.pack('<' + 'i' * len(shape), *shape) if shape else b''
        _write_array_bytes(f, shape_bytes)
        # compressMethod (empty => no compression)
        _write_string(f, "")

        # binary data (no compression): write compressedSize = raw byte length
        data_bytes = array.tobytes(order='C')
        f.write(struct.pack('<Q', len(data_bytes)))
        if len(data_bytes):
            f.write(data_bytes)

        # index: offsets then count
        for off in offsets:
            f.write(struct.pack('<Q', off))
        f.write(struct.pack('<I', len(offsets)))


def load_yt(path: str, name: Optional[str] = None) -> np.ndarray:
    """
    Load a tensor from a .yt file written by save_yt or compatible C++ writer.
    If name is None, the first tensor is returned.
    """
    with open(path, 'rb') as f:
        # header
        magic = f.read(8)
        if magic != _MAGIC:
            raise ValueError(f"Bad magic: {magic!r}")
        (version,) = struct.unpack('<B', f.read(1))
        if version != _VERSION:
            # allow but warn
            print(f"Warning: file version {version} != expected {_VERSION}")
        # metadata
        _ = _read_string(f)

        # read index: last 4 bytes is uint32 count
        f.seek(-4, 2)
        (tensor_count,) = struct.unpack('<I', f.read(4))
        index_size = 4 + tensor_count * 8
        f.seek(-index_size, 2)
        offsets = []
        for _ in range(tensor_count):
            (off,) = struct.unpack('<Q', f.read(8))
            offsets.append(off)

        # find desired tensor (name or first)
        target_idx = 0
        if name is not None:
            found = False
            for i, off in enumerate(offsets):
                f.seek(off, 0)
                tname = _read_string(f)
                if tname == name:
                    target_idx = i
                    found = True
                    break
            if not found:
                raise KeyError(f"Tensor named '{name}' not found")

        # parse tensor metadata at offsets[target_idx]
        f.seek(offsets[target_idx], 0)
        tname = _read_string(f)
        type_name = _read_string(f)
        (type_size,) = struct.unpack('<i', f.read(4))
        tensor_type = _read_string(f)
        shape_bytes = _read_array_bytes(f)
        # interpret shape_bytes as int32 little-endian
        shape = []
        if len(shape_bytes) > 0:
            count = len(shape_bytes) // 4
            shape = list(struct.unpack('<' + 'i' * count, shape_bytes))
        # compressMethod
        compress_method = _read_string(f)
        (compressed_size,) = struct.unpack('<Q', f.read(8))
        data = f.read(compressed_size)

        # since compressMethod is empty we stored raw bytes
        if compress_method != "":
            raise NotImplementedError("Only empty (no compression) is supported by this converter")

        # reconstruct numpy array
        # choose numpy dtype from type_name; fall back to constructing from itemsize
        try:
            dt = np.dtype(type_name)
            if dt.itemsize != type_size:
                # fallback: build dtype with same itemsize as recorded
                dt = np.dtype((dt.kind, type_size))
        except Exception:
            # unknown type name: assume raw bytes and construct with itemsize
            dt = np.dtype(('V', type_size))

        arr = np.frombuffer(data, dtype=dt)
        if shape:
            arr = arr.reshape(tuple(shape), order='C')
        else:
            # scalar or empty
            arr = arr.reshape(()) if arr.size == 1 else arr
        return arr


if __name__ == '__main__':
    # small demo
    a = np.arange(24, dtype=np.float32).reshape((4,6))
    save_yt('demo.yt', a, name='demo')
    b = load_yt('demo.yt', 'demo')
    print('equal:', np.array_equal(a, b))
    print('shape:', b.shape, 'dtype:', b.dtype)

    # demo: use YTFile to write multiple arrays
    x = np.arange(6, dtype=np.int32)
    y = np.linspace(0, 1, 9, dtype=np.float64).reshape((3,3,1))
    with YTFile('demo.yt', mode='w') as yf:
        yf.add('ints', x)
        yf.add('grid', y)

    # read back
    with YTFile('demo.yt', mode='r') as yf:
        names = yf.get_tensor_names()
        print('tensors in file:', names)
        xi = yf.load('ints')
        yi = yf.load('grid')
        print('ints equal:', np.array_equal(x, xi))
        print('grid equal:', np.array_equal(y, yi))
    
