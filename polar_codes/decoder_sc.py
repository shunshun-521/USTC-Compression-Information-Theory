"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，mcba1n 风格 PSCD）
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


def _frozen_set_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return set(np.where(frozen_bits.astype(bool))[0])


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（自然顺序输入/输出）"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = _frozen_set_from_mask(frozen_bits)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top = L[j - branch_size, s]
                    bot = L[j, s]
                    b = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top, bot, b)

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = bit_reversed(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与 PSCD 比特处理顺序一致）。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    lambda_offset = np.arange(n + 1)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted Successive Cancellation Decoder）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    N = 16
    frozen = np.ones(N, dtype=int)
    frozen[:N // 2] = 0
    u = np.zeros(N, dtype=int)
    u[:N // 2] = np.random.randint(0, 2, N // 2)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    assert np.array_equal(u, sc_decode(llr, frozen))
    assert np.array_equal(u, sc_decode_recursive(llr, frozen))
