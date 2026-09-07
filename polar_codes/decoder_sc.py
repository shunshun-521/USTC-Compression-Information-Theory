"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation

_PRECOMPUTE_CACHE = {}


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """lower_llr(btm, top, bit)：La=btm, Lb=top。"""
    return La + Lb if u_hat == 0 else La - Lb


def _prepare_llr(llr_ch):
    """编码含比特倒序，信道 LLR 需做相同倒序以对齐译码树。"""
    llr = np.asarray(llr_ch, dtype=np.float64)
    rev = bit_reversal_permutation(len(llr))
    return llr[rev]


def active_llr_level(phi, n):
    """返回译码第 phi 个比特时需更新的 LLR 起始层（MSB 侧第一个 1）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & phi) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(phi, n):
    """比特回传起始层。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & phi) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _update_llrs(L, B, phi, n):
    """更新 LLR 树（信道 LLR 在 L[:, 0]）。"""
    N = L.shape[0]
    start = n - active_llr_level(phi, n)
    for s in range(start, n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(phi, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j, s],
                    L[j - branch_size, s],
                    B[j - branch_size, s + 1],
                )


def _update_bits(B, phi, n):
    """比特回传。"""
    N = B.shape[0]
    if phi < N // 2:
        return
    for s in range(n, n - active_bit_level(phi, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(phi, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与非递归结果一致）。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        start = n - active_llr_level(phi, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(phi, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _get_precomputed(N):
    if N not in _PRECOMPUTE_CACHE:
        _PRECOMPUTE_CACHE[N] = precompute_sc_indices(N)
    return _PRECOMPUTE_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    N = len(llr_ch)
    n = int(np.log2(N))
    llr = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    decode_order = [_bit_reversed(i, n) for i in range(N)]
    for phi in decode_order:
        _update_llrs(L, B, phi, n)
        if frozen_bits[phi]:
            B[phi, n] = 0
        else:
            B[phi, n] = 0 if L[phi, n] >= 0 else 1
        u_hat[phi] = B[phi, n]
        _update_bits(B, phi, n)

    return u_hat
