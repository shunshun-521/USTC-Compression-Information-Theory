"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与非递归版本（基于 permuted SC，见 Vangala et al.）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """f 运算（对数域精确形式）。"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    """g 运算（l1 为下支路，l2 为上支路）。"""
    if b == 0:
        return l1 + l2
    return l1 - l2


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = _upper_llr(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)

    def decode_node(llr_node, bit_offset):
        n_len = len(llr_node)
        if n_len == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n_len // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(np.asarray(llr, dtype=np.float64), 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

_SC_CACHE = {}


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 等扩展使用）。"""
    if N in _SC_CACHE:
        return _SC_CACHE[N]
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (layer - 1))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr, layers_bit = [], []
        p = phi
        for layer in range(n):
            if p % 2 == 0:
                layers_llr.append(layer)
                p //= 2
            else:
                break
        p = phi
        for layer in range(n):
            if p % 2 == 1:
                layers_bit.append(layer)
                p //= 2
            else:
                break
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)
    result = (lambda_offset, llr_layer_vec, bit_layer_vec)
    _SC_CACHE[N] = result
    return result


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（按比特倒序调度，对数域 f/g 更新）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [_bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)
