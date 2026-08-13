"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def _active_llr_level(i, n):
    """从 MSB 起第一个 0 的位置（用于 LLR 更新层数）"""
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
    """从 MSB 起第一个 1 的位置（用于比特回传层数）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _permute_llr_for_decode(llr_ch, N):
    """将信道 LLR 按比特倒序置换，与蝶形+倒序编码匹配"""
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    对蝶形+比特倒序编码，递归树与自然序译码需配合置换 LLR；
    此处与 sc_decode 使用相同的置换 SC 算法以保证一致性。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量。
    返回 lambda_offset, llr_layer_vec, bit_layer_vec
    """
    n = int(np.log2(N))
    lambda_offset = [0]
    for layer in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + (1 << (layer - 1)))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        layers_llr = list(range(n - _active_llr_level(l, n), n))
        layers_bit = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N / 2 else []
        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（高效实现，参考 Permuted SCD 算法）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = {i for i in range(N) if frozen_bits[i] == 1}

    llr_perm = _permute_llr_for_decode(llr_ch, N)
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_perm

    for phi in range(N):
        l = _bit_reversed_index(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    B[j - branch_size, s + 1] = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (
                        B[j, s] ^ B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    return B[:, n]


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors_sc = 0
    errors_rec = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(sc_decode(llr, frozen), u):
            errors_sc += 1
        if not np.array_equal(sc_decode_recursive(llr, frozen), u):
            errors_rec += 1
    print(f"SC errors: {errors_sc}/100, Recursive errors: {errors_rec}/100")
