"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与基于惰性 LLR 计算的非递归高效实现
"""
import math
import numpy as np

_INF = np.float64(1e100)
_NEG_INF = np.float64(-1e100)

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 惰性 LLR SC（主实现）====================


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, bits, n):
    if _b_check(layer - 1, idx):
        bits[layer][idx] = bits[layer - 1][idx]
    else:
        if bits[layer - 1][idx] == -1:
            _s_updater(layer - 1, idx, bits, n)
        sibling = idx + (1 << (layer - 1))
        if bits[layer - 1][sibling] == -1:
            _s_updater(layer - 1, sibling, bits, n)
        bits[layer][idx] = bits[layer - 1][idx] ^ bits[layer - 1][sibling]


def _li(layer, idx, llrs, bits, n):
    if llrs[layer][idx] > _NEG_INF / 2:
        return llrs[layer][idx]
    if _b_check(layer, idx) == 0:
        left = _li(layer + 1, idx, llrs, bits, n)
        right = _li(layer + 1, idx + (1 << layer), llrs, bits, n)
        llrs[layer][idx] = f_operation(left, right)
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), bits, n)
        left = _li(layer + 1, idx - (1 << layer), llrs, bits, n)
        right = _li(layer + 1, idx, llrs, bits, n)
        u_partial = bits[layer][idx - (1 << layer)]
        llrs[layer][idx] = g_operation(left, right, u_partial)
    return llrs[layer][idx]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（惰性 LLR 计算）。

    frozen_bits: 1/True 表示冻结位，0/False 表示信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llrs = np.full((n + 1, N), _NEG_INF, dtype=np.float64)
    llrs[n, :] = llr_ch
    bits = -np.ones((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        if frozen_bits[phi]:
            u_hat[phi] = 0
            llrs[0][phi] = _INF
            bits[0][phi] = 0
        else:
            llr_val = _li(0, phi, llrs, bits, n)
            u_hat[phi] = 1 if llr_val < 0 else 0
            bits[0][phi] = u_hat[phi]

    return u_hat.astype(int)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(node_llr, block_len, offset):
        if block_len == 1:
            if frozen_bits[offset]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if node_llr[0] >= 0 else 1
            return
        half = block_len // 2
        llr_left = f_operation(node_llr[:half], node_llr[half:])
        decode_block(llr_left, half, offset)
        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(node_llr[:half], node_llr[half:], u_left)
        decode_block(llr_right, half, offset + half)

    decode_block(llr, N, 0)
    return u_hat


def precompute_sc_indices(N):
    """保留接口：惰性 SC 不需要预计算，返回占位结构"""
    n = int(math.log2(N))
    empty = [[] for _ in range(N)]
    lambda_offset = [1 << (n - layer) for layer in range(n + 1)]
    return lambda_offset, empty, empty


def sc_decode_layered(llr_ch, frozen_bits):
    """与 sc_decode 相同（别名）"""
    return sc_decode(llr_ch, frozen_bits)
