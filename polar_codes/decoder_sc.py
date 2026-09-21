"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，mcba1n 风格）
"""
import math

import numpy as np

from encoder import bit_reversed


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    """对数域精确 f 运算（SC 主实现使用）"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, bit):
    return (l1 + l2) if bit == 0 else (l1 - l2)


def _to_frozen_set(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(fb)[0])


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，自然信道 LLR 顺序）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_set = _to_frozen_set(frozen_bits)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_node(L, layer, idx):
        if layer == n:
            if idx in frozen_set:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if L[0] >= 0 else 1
            return

        half = 2 ** (n - layer - 1)
        L_left = np.array([
            _upper_llr_exact(L[i], L[i + half]) for i in range(half)
        ])
        decode_node(L_left, layer + 1, idx)

        u_left = u_hat[idx:idx + half]
        L_right = g_operation(L[:half], L[half:], u_left)
        decode_node(L_right, layer + 1, idx + half)

    decode_node(llr.copy(), 0, 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助信息。
    返回比特倒序译码顺序及每层活跃级别函数参数。
    """
    n = int(math.log2(N))
    decode_order = [bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = [_active_llr_level(l, n) for l in decode_order]
    bit_layer_vec = [_active_bit_level(l, n) for l in decode_order]
    return decode_order, llr_layer_vec, bit_layer_vec


_SC_PRECOMPUTE = {}


def _get_sc_precompute(N):
    if N not in _SC_PRECOMPUTE:
        _SC_PRECOMPUTE[N] = precompute_sc_indices(N)
    return _SC_PRECOMPUTE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（mcba1n 风格，比特倒序译码顺序）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_set = _to_frozen_set(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    C = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    decode_order, _, _ = _get_sc_precompute(N)

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = 0 if np.isnan(C[j - branch_size, s + 1]) else int(
                        C[j - branch_size, s + 1]
                    )
                    L[j, s + 1] = _lower_llr_exact(
                        L[j, s], L[j - branch_size, s], top_bit
                    )

        if l in frozen_set:
            C[l, n] = 0
        else:
            C[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        C[j - branch_size, s - 1] = int(C[j, s]) ^ int(
                            C[j - branch_size, s]
                        )
                        C[j, s - 1] = C[j, s]

    return C[:, n].astype(int)


def sc_decode_minsum(llr_ch, frozen_bits):
    """使用 min-sum f 运算的 SC 译码（与 f_operation 一致）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_set = _to_frozen_set(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    C = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch
    decode_order, _, _ = _get_sc_precompute(N)

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = 0 if np.isnan(C[j - branch_size, s + 1]) else int(
                        C[j - branch_size, s + 1]
                    )
                    L[j, s + 1] = _lower_llr_exact(
                        L[j, s], L[j - branch_size, s], top_bit
                    )

        if l in frozen_set:
            C[l, n] = 0
        else:
            C[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        C[j - branch_size, s - 1] = int(C[j, s]) ^ int(
                            C[j - branch_size, s]
                        )
                        C[j, s - 1] = C[j, s]

    return C[:, n].astype(int)
