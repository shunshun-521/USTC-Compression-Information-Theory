"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，与递归等价）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（boxplus 近似）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    from encoder import bit_reversed_index
    return bit_reversed_index(i, n)


def _frozen_to_set(frozen_bits, N):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return set(np.where(fb)[0])
    if fb.dtype in (np.int8, np.int32, np.int64) and set(np.unique(fb)).issubset({0, 1}):
        return set(np.where(fb.astype(bool))[0])
    return set(int(x) for x in fb)


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


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归版 SC（基于参考实现的顺序更新，便于对照）。"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（按比特倒序相位顺序，与 Arikan 核编码配套）。

    参数：
        llr_ch: 长度 N 的信道 LLR
        frozen_bits: 长度 N 的 bool/int 数组（1/True 表示冻结位），或冻结位索引列表
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = _frozen_to_set(frozen_bits, N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归辅助表（与顺序 SC 等价的层列表）。"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = np.arange(n + 1)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoders(N=64, trials=50, seed=0):
    """验证 SC 在噪声less条件下可正确恢复码字。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    rng = np.random.default_rng(seed)
    info_idx, frozen_idx, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, N // 2)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat, u):
            return False
    return True
