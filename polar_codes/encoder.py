"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
from polar_core import bit_reversal_permutation, build_generator_matrix, polar_encode

__all__ = ["polar_encode", "bit_reversal_permutation", "build_generator_matrix"]
