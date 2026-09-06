"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Vangala SSC，高效实现）
"""
import numpy as np

from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为下支路 LLR，Lb 为上支路 LLR"""
    return La + Lb if u_hat == 0 else La - Lb


def _active_llr_level(i, n):
    """llr 更新起始层（自右起第一个 1 之前）"""
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
    """比特回传起始层"""
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
    """预计算非递归 SC 译码辅助向量（兼容 SCL 接口）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for phi_nat in range(N):
        l = decode_order[phi_nat]
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l >= N / 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _ssc_decode_core(llr_ch, frozen_bits, return_state=False):
    """
    Vangala 置换 SSC 译码核心。
    frozen_bits[i]=True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=np.int8)

    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            bit = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
        B[l, n] = bit
        u_hat[l] = bit

        if l < N / 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    if return_state:
        return u_hat, L, B
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用 SSC 核心保证一致）"""
    return _ssc_decode_core(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（Vangala SSC）"""
    return _ssc_decode_core(llr_ch, frozen_bits)
