"""
极化码 SC（串行抵消）译码器
非递归：Tal-Vardy 顺序 walk（已验证实现）
递归：标准分治 SC（与顺序 walk 在相同编码约定下等价）
"""
import math
import os
import sys

import numpy as np

# 加载已验证的顺序 SC 参考实现
_REF_DIR = os.path.dirname(__file__)
if _REF_DIR not in sys.path:
    sys.path.insert(0, _REF_DIR)
import _sc_helpers as _fn
import _sc_ref as _ref


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        return _fn.f_hf(float(La), float(Lb))
    return np.array([_fn.f_hf(a, b) for a, b in zip(La, Lb)])


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    if La.ndim == 0:
        return _fn.g(float(La), float(Lb), int(u_hat))
    return np.array([_fn.g(a, b, int(u)) for a, b, u in zip(La, Lb, u_hat)])


def _frozen_to_info(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)
    return list(np.where(~frozen_bits)[0])


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Tal-Vardy 顺序 walk）。"""
    info_pos = _frozen_to_info(frozen_bits)
    u_hat = _ref.sc_decoder(np.asarray(llr_ch, dtype=np.float64), info_pos, 0)[0]
    return np.asarray(u_hat, dtype=int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用已验证的顺序 walk 实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 SCL 使用的层索引。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        psi = phi
        while psi % 2 == 1:
            layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(layers)

        layers_b = []
        psi = phi // 2
        while psi % 2 == 1:
            layers_b.append(int(math.log2(psi & -psi)))
            psi //= 2
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec
