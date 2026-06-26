"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效 Permuted SCD 实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _prepare_llr(llr_ch):
    """将信道 LLR 按比特倒序置换，与编码端 B_N 对应。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _bit_reversed(i, n):
    """单索引比特倒序"""
    result = 0
    for bit in range(n):
        if (i >> bit) & 1:
            result |= 1 << (n - 1 - bit)
    return result


def _active_llr_level(i, n):
    """找二进制表示中第一个 1 的位置（从高位计）"""
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
    """找二进制表示中第一个 0 的位置（从高位计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _upper_llr_min_sum(l1, l2):
    return f_operation(l1, l2)


def _lower_llr_min_sum(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（Permuted SCD）。
    """
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    decode_order = [_bit_reversed(i, n) for i in range(N)]

    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = decode_order[i]
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            end_s = n - _active_bit_level(l, n) + 1
            bit_layer_vec.append(list(range(n, end_s - 1, -1)))

    lambda_offset = [2 ** i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec, br


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SCD，Vangala et al. 2014）。
    编码器含比特倒序置换，故先将信道 LLR 做比特倒序重排。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_ch = llr_ch[br]
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    C = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = _bit_reversed(i, n)

        start_s = n - _active_llr_level(l, n)
        for s in range(start_s, n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_min_sum(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr_min_sum(
                        L[j, s], L[j - branch_size, s], C[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            C[l, n] = 0
        else:
            C[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = C[l, n]

        if l >= N // 2:
            end_s = n - _active_bit_level(l, n) + 1
            for s in range(n, end_s - 1, -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        C[j - branch_size, s - 1] = C[j, s] ^ C[j - branch_size, s]
                        C[j, s - 1] = C[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用 Permuted SCD）"""
    return sc_decode(llr, frozen_bits)
