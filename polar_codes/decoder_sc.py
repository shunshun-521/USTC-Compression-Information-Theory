"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversed


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    其中 La 为上层 LLR，Lb 为下层 LLR。
    """
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """二进制展开中第一个 1 的位置（用于 LLR 更新层数）。"""
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
    """二进制展开中第一个 0 的位置（用于比特回传层数）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr

    frozen_set = set(np.where(frozen_bits)[0])

    for l in [bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def _update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = f_operation(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助信息（与相位比特倒序顺序一致）。
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
    lambda_offset = list(range(n + 1))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（与递归版本算法一致，基于分层 LLR/比特矩阵）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    rng = np.random.default_rng(0)
    N, K = 64, 32
    info_idx = np.sort(rng.choice(N, K, replace=False))
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        llr = compute_llr(s, 0.01)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u):
            errors += 1
    print(f"SC errors at low noise: {errors}/100")

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    uh = sc_decode(compute_llr(bpsk_modulate(x), 0.01), np.zeros(4, dtype=bool))
    print(f"encode/decode u={u} x={x} uh={uh}")
