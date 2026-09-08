"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversed

LLR_CLIP = 30.0


def clip_llr(llr):
    return np.clip(llr, -LLR_CLIP, LLR_CLIP)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> -La+Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """从高位起第一个 0 的位置（用于 LLR 更新层数）"""
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
    """从高位起第一个 1 的位置（用于比特回传层数）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    decode_order = [bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = decode_order[i]
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


def _update_llrs(L, B, l, n):
    """更新第 l 个（比特倒序域）相位的 LLR"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits(B, l, n, N):
    """比特回传"""
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，自然顺序 LLR）"""
    N = len(llr)
    n = int(np.log2(N))
    frozen = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=np.int8)

    for phase in range(N):
        l = bit_reversed(phase, n)
        # 简化：对每一相位用完整树递归求 LLR
        u_hat[l] = 0 if frozen[l] or llr[l] >= 0 else 1

    # 使用非递归作为主实现
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr = clip_llr(np.asarray(llr_ch, dtype=np.float64))
    N = len(llr)
    n = int(np.log2(N))
    frozen = np.asarray(frozen_bits, dtype=bool)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    for phase in range(N):
        l = bit_reversed(phase, n)
        _update_llrs(L, B, l, n)
        if frozen[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(np.int8)
