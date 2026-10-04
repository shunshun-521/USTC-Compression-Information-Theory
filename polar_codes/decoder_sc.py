"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(bool) if fb.dtype == bool else (fb != 0)


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


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """f 运算（boxplus，对数域稳定实现）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    vec = np.vectorize(
        lambda a, b: _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b), otypes=[float]
    )
    return vec(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    flip = np.where(u_hat.astype(int) == 0, 1.0, -1.0)
    if La.ndim == 0 and Lb.ndim == 0:
        return float(flip) * float(La) + float(Lb)
    return flip * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            if frozen[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_u = f_operation(llr_node[:half], llr_node[half:])
        decode_block(llr_u, offset)
        u_left = u_hat[offset : offset + half]
        llr_v = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_block(llr_v, offset + half)

    decode_block(llr, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD，O(N log N)）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    def upper_llr(l1, l2):
        if np.isinf(l1) and not np.isinf(l2):
            return l2
        if not np.isinf(l1) and np.isinf(l2):
            return l1
        if np.isinf(l1) and np.isinf(l2):
            return np.inf
        return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)

    def lower_llr(l1, l2, b):
        if b == 0:
            if np.isinf(l1) or np.isinf(l2):
                return np.inf
            return l1 + l2
        return l1 - l2

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = _bit_reversed(i, n)
        update_llrs(l)
        if frozen[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi % 2 == 0:
            bit_layers.append(0)
        t = phi >> 1
        l = 1
        while t > 0:
            if t % 2 == 0:
                bit_layers.append(l)
            t >>= 1
            l += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
