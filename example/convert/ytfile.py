"""
YTFile - 独立的 .yt 文件读写模块

此模块提供与 C++ YTensorIO 格式兼容的 Python 读写接口。
可以被其他转换器模块 import 使用。

File format:
- Magic: "YTENSORF" (8 bytes)
- Version: uint8
- Metadata: string
- Tensors: [name, type_name, type_size, tensor_type, shape, compress_method, compressed_size, data]...
- Index: [offset1, offset2, ...] + tensor_count (uint32)
"""

import struct
import numpy as np
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

__all__ = ['YTFile', 'save_yt', 'load_yt', 'MAGIC', 'VERSION']

MAGIC = b"YTENSORF"  # 8 bytes
VERSION = 0


# ============ Binary Format Helpers ============

def _write_string(f, s: str) -> None:
    """Write a length-prefixed UTF-8 string."""
    b = s.encode('utf-8')
    f.write(struct.pack('<I', len(b)))
    if len(b):
        f.write(b)


def _read_string(f) -> str:
    """Read a length-prefixed UTF-8 string."""
    (length,) = struct.unpack('<I', f.read(4))
    if length == 0:
        return ""
    b = f.read(length)
    return b.decode('utf-8')


def _write_array_bytes(f, data_bytes: bytes) -> None:
    """Write byte array with uint64 length prefix."""
    f.write(struct.pack('<Q', len(data_bytes)))
    if len(data_bytes):
        f.write(data_bytes)


def _read_array_bytes(f) -> bytes:
    """Read byte array with uint64 length prefix."""
    (count,) = struct.unpack('<Q', f.read(8))
    if count == 0:
        return b''
    return f.read(count)


# ============ Type Mapping ============

# NumPy dtype name -> YTensor type name mapping
NUMPY_TO_YTENSOR_TYPE = {
    'float16': 'float16',
    'float32': 'float32',
    'float64': 'float64',
    'int8': 'int8',
    'int16': 'int16',
    'int32': 'int32',
    'int64': 'int64',
    'uint8': 'uint8',
    'uint16': 'uint16',
    'uint32': 'uint32',
    'uint64': 'uint64',
    'bool': 'bool',
    'bfloat16': 'bfloat16',
}


def numpy_dtype_to_ytensor_type(dtype: np.dtype) -> str:
    """Convert numpy dtype to YTensor type name string."""
    name = dtype.name
    if name in NUMPY_TO_YTENSOR_TYPE:
        return NUMPY_TO_YTENSOR_TYPE[name]
    # fallback: return original name
    return name


def ytensor_type_to_numpy_dtype(type_name: str, type_size: int) -> np.dtype:
    """Convert YTensor type name to numpy dtype."""
    try:
        dt = np.dtype(type_name)
        if dt.itemsize != type_size:
            dt = np.dtype((dt.kind, type_size))
        return dt
    except TypeError:
        # unknown type, create void type with correct size
        return np.dtype(('V', type_size))


# ============ YTFile Class ============

class YTFile:
    """
    Manage multiple numpy arrays in a single .yt file.

    Usage (write):
        with YTFile('data.yt', mode='w') as ytf:
            ytf.add('tensor_a', arr1)
            ytf.add('tensor_b', arr2)

    Usage (read):
        with YTFile('data.yt', mode='r') as ytf:
            print(ytf.get_tensor_names())
            arr = ytf.load('tensor_a')

    Usage (append):
        with YTFile('data.yt', mode='a') as ytf:
            ytf.add('tensor_c', arr3)  # adds to existing file

    Modes:
        'w' - create new file (overwrite if exists)
        'a' - append to existing file (reads existing index, rewrites on close)
        'r' - read-only (can list and load tensors)
    """

    def __init__(self, path: Union[str, Path], mode: str = 'w'):
        """
        Initialize YTFile.
        
        Args:
            path: Path to the .yt file
            mode: 'w' (write), 'a' (append), or 'r' (read)
        """
        if mode not in ('w', 'a', 'r'):
            raise ValueError(f"Invalid mode: {mode}. Must be 'w', 'a', or 'r'")
        
        self.path = str(path)
        self.mode = mode
        self._infos: List[Dict[str, Any]] = []
        self._closed = False

        # if opening existing file, read headers and index
        if mode in ('a', 'r'):
            self._load_existing()

    def _load_existing(self) -> None:
        """Load tensor metadata from existing file."""
        try:
            with open(self.path, 'rb') as f:
                magic = f.read(8)
                if magic != MAGIC:
                    raise ValueError(f'Invalid file format: bad magic {magic!r}')
                
                (version,) = struct.unpack('<B', f.read(1))
                _ = _read_string(f)  # metadata (currently unused)
                
                # read index from end of file
                f.seek(-4, 2)
                (tensor_count,) = struct.unpack('<I', f.read(4))
                index_size = 4 + tensor_count * 8
                f.seek(-index_size, 2)
                offsets = [struct.unpack('<Q', f.read(8))[0] for _ in range(tensor_count)]
                
                # read each tensor header
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
                    
                    self._infos.append({
                        'name': name,
                        'type_name': type_name,
                        'type_size': type_size,
                        'tensor_type': tensor_type,
                        'shape': shape,
                        'compress_method': compress_method,
                        'compressed_size': compressed_size,
                        'data_offset': data_offset
                    })
        except FileNotFoundError:
            if self.mode == 'r':
                raise FileNotFoundError(f"File not found: {self.path}")
            # append mode: no existing file is OK

    def add(self, name: str, array: np.ndarray, tensor_type: str = 'dense') -> None:
        """
        Stage a numpy array to be written into the file on close.
        
        Args:
            name: Tensor name (must be unique within file)
            array: NumPy array to store
            tensor_type: Type string (default 'dense')
        """
        if self._closed:
            raise RuntimeError("Cannot add to closed YTFile")
        if self.mode == 'r':
            raise RuntimeError("Cannot add tensors in read mode")
        
        dtype = array.dtype
        type_name = numpy_dtype_to_ytensor_type(dtype)
        type_size = dtype.itemsize
        shape = [int(s) for s in array.shape]
        data_bytes = array.tobytes(order='C')
        
        info = {
            'name': name,
            'type_name': type_name,
            'type_size': type_size,
            'tensor_type': tensor_type,
            'shape': shape,
            'compress_method': '',
            'compressed_size': len(data_bytes),
            'data': data_bytes
        }
        
        # if existing name, overwrite
        for i, e in enumerate(self._infos):
            if e['name'] == name:
                self._infos[i] = info
                return
        self._infos.append(info)

    def add_raw(self, name: str, data: bytes, type_name: str, type_size: int, 
                shape: List[int], tensor_type: str = 'dense') -> None:
        """
        Add raw bytes directly without numpy conversion.
        Useful for format converters that handle raw data.
        
        Args:
            name: Tensor name
            data: Raw bytes (C-order)
            type_name: Type name string (e.g., 'float32')
            type_size: Size of each element in bytes
            shape: List of dimensions
            tensor_type: Type string (default 'dense')
        """
        if self._closed:
            raise RuntimeError("Cannot add to closed YTFile")
        if self.mode == 'r':
            raise RuntimeError("Cannot add tensors in read mode")
        
        info = {
            'name': name,
            'type_name': type_name,
            'type_size': type_size,
            'tensor_type': tensor_type,
            'shape': shape,
            'compress_method': '',
            'compressed_size': len(data),
            'data': data
        }
        
        # if existing name, overwrite
        for i, e in enumerate(self._infos):
            if e['name'] == name:
                self._infos[i] = info
                return
        self._infos.append(info)

    def get_tensor_names(self) -> List[str]:
        """Get list of all tensor names in the file."""
        return [e['name'] for e in self._infos]

    def get_tensor_info(self, name: str) -> Dict[str, Any]:
        """Get metadata for a specific tensor."""
        for e in self._infos:
            if e['name'] == name:
                return {
                    'name': e['name'],
                    'type_name': e['type_name'],
                    'type_size': e['type_size'],
                    'tensor_type': e['tensor_type'],
                    'shape': e['shape'],
                    'compress_method': e['compress_method'],
                    'size_bytes': e['compressed_size']
                }
        raise KeyError(f"Tensor named '{name}' not found")

    def load(self, name: Optional[str] = None) -> np.ndarray:
        """
        Load a tensor by name.
        
        Args:
            name: Tensor name. If None, loads the first tensor.
            
        Returns:
            NumPy array with the tensor data.
        """
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
            raise NotImplementedError('Compression not supported')
        
        dt = ytensor_type_to_numpy_dtype(info['type_name'], info['type_size'])
        arr = np.frombuffer(data, dtype=dt)
        
        if info['shape']:
            arr = arr.reshape(tuple(info['shape']), order='C')
        return arr

    def load_raw(self, name: str) -> tuple:
        """
        Load raw bytes and metadata for a tensor.
        
        Returns:
            Tuple of (data_bytes, type_name, type_size, shape, tensor_type)
        """
        for info in self._infos:
            if info['name'] == name:
                if 'data' in info:
                    data = info['data']
                else:
                    with open(self.path, 'rb') as f:
                        f.seek(info['data_offset'], 0)
                        data = f.read(info['compressed_size'])
                return (data, info['type_name'], info['type_size'], 
                        info['shape'], info['tensor_type'])
        raise KeyError(f"Tensor named '{name}' not found")

    def save_all(self) -> None:
        """Write all staged (and existing) tensors into file."""
        if self.mode == 'r':
            raise RuntimeError("Cannot save in read mode")
        
        # Prepare buffers for any entries without in-memory 'data'
        for e in self._infos:
            if 'data' not in e:
                with open(self.path, 'rb') as f:
                    f.seek(e['data_offset'], 0)
                    e['data'] = f.read(e['compressed_size'])

        # Write new file
        with open(self.path, 'wb') as f:
            # header
            f.write(MAGIC)
            f.write(struct.pack('<B', VERSION))
            _write_string(f, '')  # metadata

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
            
            # index
            for off in offsets:
                f.write(struct.pack('<Q', off))
            f.write(struct.pack('<I', len(offsets)))

    def close(self) -> None:
        """Close the file, writing changes if in write/append mode."""
        if not self._closed:
            if self.mode in ('w', 'a'):
                self.save_all()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __len__(self) -> int:
        return len(self._infos)

    def __contains__(self, name: str) -> bool:
        return any(e['name'] == name for e in self._infos)

    def __iter__(self):
        return iter(self.get_tensor_names())


# ============ Convenience Functions ============

def save_yt(path: Union[str, Path], array: np.ndarray, name: str = '0') -> None:
    """
    Save a single numpy ndarray to a .yt file.
    
    Args:
        path: Output file path
        array: NumPy array to save
        name: Tensor name (default '0')
    """
    with YTFile(path, mode='w') as ytf:
        ytf.add(name, array)


def load_yt(path: Union[str, Path], name: Optional[str] = None) -> np.ndarray:
    """
    Load a tensor from a .yt file.
    
    Args:
        path: Input file path
        name: Tensor name. If None, loads the first tensor.
        
    Returns:
        NumPy array
    """
    with YTFile(path, mode='r') as ytf:
        return ytf.load(name)


# ============ Test / Demo ============

if __name__ == '__main__':
    import tempfile
    import os
    
    print("=== YTFile Module Test ===\n")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, 'test.yt')
        
        # Test 1: Single tensor
        print("Test 1: Single tensor save/load")
        a = np.arange(24, dtype=np.float32).reshape((4, 6))
        save_yt(test_path, a, name='demo')
        b = load_yt(test_path, 'demo')
        assert np.array_equal(a, b), "Single tensor test failed"
        print(f"  Shape: {b.shape}, dtype: {b.dtype} - PASS\n")
        
        # Test 2: Multiple tensors
        print("Test 2: Multiple tensors")
        x = np.arange(6, dtype=np.int32)
        y = np.linspace(0, 1, 9, dtype=np.float64).reshape((3, 3))
        z = np.random.randn(2, 3, 4).astype(np.float32)
        
        with YTFile(test_path, mode='w') as yf:
            yf.add('ints', x)
            yf.add('grid', y)
            yf.add('random', z)
        
        with YTFile(test_path, mode='r') as yf:
            names = yf.get_tensor_names()
            print(f"  Tensors in file: {names}")
            assert names == ['ints', 'grid', 'random'], "Name list mismatch"
            
            xi = yf.load('ints')
            yi = yf.load('grid')
            zi = yf.load('random')
            assert np.array_equal(x, xi), "ints mismatch"
            assert np.array_equal(y, yi), "grid mismatch"
            assert np.array_equal(z, zi), "random mismatch"
            print("  All tensors match - PASS\n")
        
        # Test 3: Append mode
        print("Test 3: Append mode")
        w = np.array([1, 2, 3], dtype=np.int64)
        with YTFile(test_path, mode='a') as yf:
            yf.add('appended', w)
        
        with YTFile(test_path, mode='r') as yf:
            names = yf.get_tensor_names()
            print(f"  Tensors after append: {names}")
            assert len(names) == 4, "Append failed"
            assert 'appended' in yf
            wa = yf.load('appended')
            assert np.array_equal(w, wa), "Appended tensor mismatch"
            print("  Append successful - PASS\n")
        
        # Test 4: Get tensor info
        print("Test 4: Tensor info")
        with YTFile(test_path, mode='r') as yf:
            info = yf.get_tensor_info('random')
            print(f"  Info for 'random': {info}")
            assert info['shape'] == [2, 3, 4]
            assert info['type_name'] == 'float32'
            print("  Info correct - PASS\n")
        
        # Test 5: Raw data API
        print("Test 5: Raw data API")
        raw_data = b'\x00\x01\x02\x03\x04\x05\x06\x07'
        with YTFile(test_path, mode='w') as yf:
            yf.add_raw('raw_tensor', raw_data, 'uint8', 1, [8])
        
        with YTFile(test_path, mode='r') as yf:
            data, type_name, type_size, shape, tensor_type = yf.load_raw('raw_tensor')
            assert data == raw_data
            assert type_name == 'uint8'
            assert shape == [8]
            print("  Raw API works - PASS\n")
    
    print("=== All Tests Passed ===")
