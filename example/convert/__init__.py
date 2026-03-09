"""
YTensor Python Converters Package

This package provides utilities for converting between various tensor formats
and the YTensor (.yt) format used by the C++ YTensor library.

Modules:
    ytfile - Core YTFile class for reading/writing .yt files
    numpy2yt - Legacy numpy to .yt converter
    safetensors2yt - HuggingFace SafeTensors to .yt converter

Usage:
    from convert import YTFile, save_yt, load_yt
    
    # Write multiple tensors
    with YTFile('data.yt', mode='w') as ytf:
        ytf.add('weights', weights_array)
        ytf.add('bias', bias_array)
    
    # Read tensors
    with YTFile('data.yt', mode='r') as ytf:
        weights = ytf.load('weights')
"""

from .ytfile import YTFile, save_yt, load_yt, MAGIC, VERSION

__all__ = ['YTFile', 'save_yt', 'load_yt', 'MAGIC', 'VERSION']
__version__ = '1.0.0'
