"""
极化码 SC（串行抵消）译码器
基于按需 LLR 计算（与经典因子图索引一致）
"""
import math
import numpy as np

_LLR_UNSET = -np.inf


def f_operation(La, Lb):
    """min-sum f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, index):
    return (index // (1 << layer)) % 2


def _s_updater(layer, index, bits):
    if _b_check(layer - 1, index):
        bits[layer, index] = bits[layer - 1, index]
    else:
        if bits[layer - 1, index] == -1:
            _s_updater(layer - 1, index, bits)
        pair = index + (1 << (layer - 1))
        if bits[layer - 1, pair] == -1:
            _s_updater(layer - 1, pair, bits)
        bits[layer, index] = bits[layer - 1, index] ^ bits[layer - 1, pair]


def _li(layer, index, llrs, bits, n):
    if llrs[layer, index] != _LLR_UNSET:
        return llrs[layer, index]

    if _b_check(layer, index) == 0:
        llrs[layer, index] = f_operation(
            _li(layer + 1, index, llrs, bits, n),
            _li(layer + 1, index + (1 << layer), llrs, bits, n),
        )
    else:
        if layer > 0:
            _s_updater(layer, index - (1 << layer), bits)
        left = index - (1 << layer)
        llrs[layer, index] = g_operation(
            _li(layer + 1, left, llrs, bits, n),
            _li(layer + 1, index, llrs, bits, n),
            bits[layer, left],
        )
    return llrs[layer, index]


def sc_decode_recursive(llr, frozen_bits):
    """SC 译码"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), _LLR_UNSET, dtype=np.float64)
    llrs[n, :] = llr
    bits = np.full((n + 1, N), -1, dtype=int)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        llr_phi = _li(0, phi, llrs, bits, n)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 1 if llr_phi < 0 else 0
        bits[0, phi] = u_hat[phi]

    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers, bit_layers = [], []
        p, layer = phi, 0
        while layer < n:
            if (p & 1) == 0:
                llr_layers.append(layer)
                p >>= 1
                layer += 1
            else:
                break
        p, layer = phi, 0
        while layer < n:
            if (p & 1) == 1:
                bit_layers.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数"""
    return sc_decode_recursive(llr_ch, frozen_bits)


# 兼容 SCL 模块
_compute_llr = _li
_LLR_UNSET = _LLR_UNSET
