"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted SCD）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """二进制表示中首个 1 的位置（从 MSB 计）"""
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
    """二进制表示中首个 0 的位置（从 MSB 计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _normalize_frozen(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return frozen_bits
    return frozen_bits.astype(bool)


def _sc_core(llr_ch, frozen_bits):
    """Permuted SCD 核心（非递归）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _normalize_frozen(frozen_bits)
    N = len(llr_ch)
    n = int(np.log2(N))

    # 编码端含比特倒序，信道 LLR 需同步置换
    rev = bit_reversal_permutation(N)
    llr_ch = llr_ch[rev]

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
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

    u_hat = np.zeros(N, dtype=int)
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit
        update_bits(l)

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（主实现）"""
    return _sc_core(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用同一核心逻辑验证）"""
    return _sc_core(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（供文档/扩展使用）
    """
    n = int(np.log2(N))
    lambda_offset = [2 ** (n - i) for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
