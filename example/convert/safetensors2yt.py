"""
SafeTensors to YT Converter

将 HuggingFace SafeTensors 格式转换为 YTensor (.yt) 格式。

Usage:
    python safetensors2yt.py model.safetensors output.yt
    python safetensors2yt.py model_dir/ output.yt  # 转换目录中所有 .safetensors 文件
    
Dependencies:
    pip install safetensors numpy

SafeTensors format reference:
    https://huggingface.co/docs/safetensors
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

# Import our YTFile module
from ytfile import YTFile

# Try to import safetensors
try:
    from safetensors import safe_open
    from safetensors.numpy import load_file as load_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors package not installed. Install with: pip install safetensors")


# SafeTensors dtype mapping
SAFETENSORS_DTYPE_MAP = {
    'F64': ('float64', 8),
    'F32': ('float32', 4),
    'F16': ('float16', 2),
    'BF16': ('bfloat16', 2),
    'I64': ('int64', 8),
    'I32': ('int32', 4),
    'I16': ('int16', 2),
    'I8': ('int8', 1),
    'U64': ('uint64', 8),
    'U32': ('uint32', 4),
    'U16': ('uint16', 2),
    'U8': ('uint8', 1),
    'BOOL': ('bool', 1),
}


def get_safetensors_metadata(path: str) -> Tuple[Dict, int]:
    """
    Read SafeTensors file header to get tensor metadata without loading data.
    
    Returns:
        Tuple of (metadata_dict, header_size)
    """
    with open(path, 'rb') as f:
        # First 8 bytes: header size (uint64 LE)
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        
        # Read header JSON
        header_json = f.read(header_size)
        metadata = json.loads(header_json.decode('utf-8'))
        
        return metadata, 8 + header_size


def convert_safetensors_to_yt(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    tensor_filter: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    """
    Convert a SafeTensors file to YT format.
    
    Args:
        input_path: Path to input .safetensors file
        output_path: Path to output .yt file
        tensor_filter: Optional list of tensor names to include (None = all)
        verbose: Print progress information
        
    Returns:
        Number of tensors converted
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors package required. Install with: pip install safetensors")
    
    input_path = str(input_path)
    output_path = str(output_path)
    
    if verbose:
        print(f"Converting: {input_path} -> {output_path}")
    
    # Load safetensors file
    tensors = load_safetensors(input_path)
    
    # Filter tensors if specified
    if tensor_filter:
        tensors = {k: v for k, v in tensors.items() if k in tensor_filter}
    
    if verbose:
        print(f"  Found {len(tensors)} tensors")
    
    # Write to YT format
    count = 0
    with YTFile(output_path, mode='w') as ytf:
        for name, array in tensors.items():
            if verbose:
                print(f"    {name}: shape={array.shape}, dtype={array.dtype}")
            
            # Handle bfloat16 specially (numpy doesn't support it natively)
            # safetensors returns it as uint16, we need to preserve the type info
            if hasattr(array, 'dtype'):
                ytf.add(name, array)
            count += 1
    
    if verbose:
        print(f"  Converted {count} tensors\n")
    
    return count


def convert_safetensors_to_yt_streaming(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    tensor_filter: Optional[List[str]] = None,
    verbose: bool = True
) -> int:
    """
    Convert a SafeTensors file to YT format using streaming (memory-efficient).
    
    This reads tensors one at a time using mmap, useful for large models.
    
    Args:
        input_path: Path to input .safetensors file
        output_path: Path to output .yt file
        tensor_filter: Optional list of tensor names to include (None = all)
        verbose: Print progress information
        
    Returns:
        Number of tensors converted
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors package required. Install with: pip install safetensors")
    
    input_path = str(input_path)
    output_path = str(output_path)
    
    if verbose:
        print(f"Converting (streaming): {input_path} -> {output_path}")
    
    count = 0
    with safe_open(input_path, framework="numpy") as sf:
        tensor_names = sf.keys()
        
        if tensor_filter:
            tensor_names = [n for n in tensor_names if n in tensor_filter]
        
        if verbose:
            print(f"  Found {len(tensor_names)} tensors")
        
        with YTFile(output_path, mode='w') as ytf:
            for name in tensor_names:
                array = sf.get_tensor(name)
                if verbose:
                    print(f"    {name}: shape={array.shape}, dtype={array.dtype}")
                ytf.add(name, array)
                count += 1
    
    if verbose:
        print(f"  Converted {count} tensors\n")
    
    return count


def convert_safetensors_directory(
    input_dir: Union[str, Path],
    output_path: Union[str, Path],
    pattern: str = "*.safetensors",
    streaming: bool = True,
    verbose: bool = True
) -> int:
    """
    Convert all SafeTensors files in a directory to a single YT file.
    
    Args:
        input_dir: Directory containing .safetensors files
        output_path: Path to output .yt file
        pattern: Glob pattern for safetensors files
        streaming: Use memory-efficient streaming mode
        verbose: Print progress information
        
    Returns:
        Total number of tensors converted
    """
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors package required. Install with: pip install safetensors")
    
    input_dir = Path(input_dir)
    output_path = str(output_path)
    
    # Find all safetensors files
    safetensor_files = sorted(input_dir.glob(pattern))
    
    if not safetensor_files:
        raise FileNotFoundError(f"No {pattern} files found in {input_dir}")
    
    if verbose:
        print(f"Found {len(safetensor_files)} safetensors files in {input_dir}")
    
    total_count = 0
    
    with YTFile(output_path, mode='w') as ytf:
        for sf_path in safetensor_files:
            if verbose:
                print(f"\nProcessing: {sf_path.name}")
            
            if streaming:
                with safe_open(str(sf_path), framework="numpy") as sf:
                    for name in sf.keys():
                        array = sf.get_tensor(name)
                        if verbose:
                            print(f"    {name}: shape={array.shape}, dtype={array.dtype}")
                        ytf.add(name, array)
                        total_count += 1
            else:
                tensors = load_safetensors(str(sf_path))
                for name, array in tensors.items():
                    if verbose:
                        print(f"    {name}: shape={array.shape}, dtype={array.dtype}")
                    ytf.add(name, array)
                    total_count += 1
    
    if verbose:
        print(f"\nTotal: Converted {total_count} tensors to {output_path}")
    
    return total_count


def list_safetensors(path: Union[str, Path]) -> None:
    """List all tensors in a SafeTensors file."""
    if not HAS_SAFETENSORS:
        raise ImportError("safetensors package required. Install with: pip install safetensors")
    
    path = str(path)
    print(f"Tensors in {path}:\n")
    
    with safe_open(path, framework="numpy") as sf:
        for name in sf.keys():
            # Get tensor without loading data
            tensor = sf.get_tensor(name)
            print(f"  {name}:")
            print(f"    shape: {tensor.shape}")
            print(f"    dtype: {tensor.dtype}")
            print(f"    size:  {tensor.nbytes / 1024 / 1024:.2f} MB")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Convert SafeTensors to YTensor (.yt) format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert a single file
    python safetensors2yt.py model.safetensors model.yt
    
    # Convert all safetensors in a directory
    python safetensors2yt.py model_dir/ all_weights.yt
    
    # List tensors in a file
    python safetensors2yt.py --list model.safetensors
    
    # Convert specific tensors only
    python safetensors2yt.py model.safetensors out.yt --filter "model.embed_tokens.weight" "lm_head.weight"
"""
    )
    
    parser.add_argument('input', type=str,
                        help='Input .safetensors file or directory')
    parser.add_argument('output', type=str, nargs='?', default=None,
                        help='Output .yt file (default: input_name.yt)')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List tensors in input file instead of converting')
    parser.add_argument('--filter', '-f', nargs='+', default=None,
                        help='Only convert specified tensor names')
    parser.add_argument('--no-stream', action='store_true',
                        help='Load entire file into memory (faster but uses more RAM)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    if not HAS_SAFETENSORS:
        print("Error: safetensors package not installed.")
        print("Install with: pip install safetensors")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)
    
    # List mode
    if args.list:
        if input_path.is_dir():
            for sf_file in sorted(input_path.glob("*.safetensors")):
                list_safetensors(sf_file)
        else:
            list_safetensors(input_path)
        return
    
    # Determine output path
    if args.output is None:
        if input_path.is_dir():
            output_path = input_path.parent / f"{input_path.name}.yt"
        else:
            output_path = input_path.with_suffix('.yt')
    else:
        output_path = Path(args.output)
    
    verbose = not args.quiet
    streaming = not args.no_stream
    
    # Convert
    try:
        if input_path.is_dir():
            count = convert_safetensors_directory(
                input_path, output_path,
                streaming=streaming,
                verbose=verbose
            )
        else:
            if streaming:
                count = convert_safetensors_to_yt_streaming(
                    input_path, output_path,
                    tensor_filter=args.filter,
                    verbose=verbose
                )
            else:
                count = convert_safetensors_to_yt(
                    input_path, output_path,
                    tensor_filter=args.filter,
                    verbose=verbose
                )
        
        if verbose:
            file_size = output_path.stat().st_size / 1024 / 1024
            print(f"Done! Output: {output_path} ({file_size:.2f} MB, {count} tensors)")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
