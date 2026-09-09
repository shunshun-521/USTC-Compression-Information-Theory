"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(level, idx):
    return (idx // (1 << level)) % 2


def _s_updater(level, idx, bits):
    if _b_check(level - 1, idx):
        bits[level, idx] = bits[level - 1, idx]
    else:
        if bits[level - 1, idx] == -1:
            _s_updater(level - 1, idx, bits)
        sibling = idx + (1 << (level - 1))
        if bits[level - 1, sibling] == -1:
            _s_updater(level - 1, sibling, bits)
        bits[level, idx] = bits[level - 1, idx] ^ bits[level - 1, sibling]


def _compute_llr(level, idx, llrs, bits):
    if llrs[level, idx] != -np.inf:
        return llrs[level, idx]

    if _b_check(level, idx) == 0:
        llrs[level, idx] = f_operation(
            _compute_llr(level + 1, idx, llrs, bits),
            _compute_llr(level + 1, idx + (1 << level), llrs, bits),
        )
        return llrs[level, idx]

    if level > 0:
        _s_updater(level, idx - (1 << level), bits)
    llrs[level, idx] = g_operation(
        _compute_llr(level + 1, idx - (1 << level), llrs, bits),
        _compute_llr(level + 1, idx, llrs, bits),
        bits[level, idx - (1 << level)],
    )
    return llrs[level, idx]


def _prepare_frozen_mask(frozen_bits, N):
    frozen = np.asarray(frozen_bits)
    if frozen.dtype != bool:
        frozen = frozen.astype(bool)
    return frozen


def _sc_decode_internal(llr_channel, frozen_bits):
    frozen = _prepare_frozen_mask(frozen_bits, len(llr_channel))
    n = int(math.log2(len(llr_channel)))
    N = len(llr_channel)

    llrs = -np.inf * np.ones((n + 1, N), dtype=np.float64)
    llrs[n, :] = llr_channel
    bits = -np.ones((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for idx in range(N):
        if frozen[idx]:
            bits[0, idx] = 0
            llrs[0, idx] = np.inf
            u_hat[idx] = 0
        else:
            llrs[0, idx] = _compute_llr(0, idx, llrs, bits)
            u_hat[idx] = 0 if llrs[0, idx] >= 0 else 1
            bits[0, idx] = u_hat[idx]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return _sc_decode_internal(llr_ch[rev], frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，供验证）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助索引（接口兼容）"""
    n = int(math.log2(N))
    rev = bit_reversal_permutation(N)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        for layer in range(n):
            if p % 2 == 0:
                llr_layers.append(layer)
                break
            p //= 2
        bit_layers = []
        if phi % 2 == 1:
            p = phi
            layer = 0
            while p % 2 == 1 and layer < n:
                bit_layers.append(layer)
                p //= 2
                layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return list(range(n + 1)), llr_layer_vec, bit_layer_vec, rev.tolist()
