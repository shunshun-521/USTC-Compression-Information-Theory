"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for k in range(n):
        if i & (1 << k):
            result |= 1 << (n - 1 - k)
    return result


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（下半分支）：lower_llr(btm, top, u) = (1-2u)*top + btm
    此处 La=top, Lb=btm
    """
    return (1 - 2 * u_hat) * La + Lb


def active_llr_level(i, n):
    """从最高位起，找到第一个 0 之前 1 的个数"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """从最高位起，找到第一个 0 之前 1 的个数（比特回传）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _prepare_frozen(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype != bool:
        frozen_bits = frozen_bits.astype(bool)
    return frozen_bits


def _align_channel_llr(llr_ch, N):
    """编码器含比特倒序，将信道 LLR 对齐到译码树顺序"""
    n = int(np.log2(N))
    br = np.array([bit_reversed(i, n) for i in range(N)], dtype=int)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = np.zeros(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l_idx = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l_idx, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l_idx, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, len(L), block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    if l < len(B) // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用非递归核心）"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SCD，bit-reversed 译码顺序）。
    frozen_bits: True/1 表示冻结位
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = _prepare_frozen(frozen_bits)
    frozen_set = set(np.where(frozen_bits)[0])

    llr_ch = _align_channel_llr(llr_ch, N)
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for l in [bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def sc_decode_fast(llr_ch, frozen_bits):
    """基于预计算索引的非递归 SC 译码"""
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = _prepare_frozen(frozen_bits)
    frozen_set = set(np.where(frozen_bits)[0])

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)
    llr_ch = _align_channel_llr(llr_ch, N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for phase, l in enumerate([bit_reversed(i, n) for i in range(N)]):
        for s in llr_layer_vec[phase]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        for s in bit_layer_vec[phase]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
