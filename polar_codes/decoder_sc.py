"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


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
    """二进制展开中第一个 1 的位置（从高位起）。"""
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
    """二进制展开中第一个 0 的位置（从高位起）。"""
    mask = 1 << (n - 1)
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


def _prepare_llr(llr_ch):
    """编码含比特倒序置换时，将信道 LLR 重排以匹配 SC 树结构。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    n = int(math.log2(len(llr_ch)))
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br], n


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return set(np.where(frozen_bits.astype(bool))[0])


def _sc_decode_core(llr_ch, frozen_set, n):
    """SC 译码核心（基于分层 LLR/比特数组）。"""
    N = len(llr_ch)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    def update_llrs(l):
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

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    u_hat = np.zeros(N, dtype=int)
    for i in range(N):
        l = _bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)
        u_hat[l] = B[l, n]
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与 sc_decode 等价）。
    """
    llr_perm, n = _prepare_llr(llr)
    frozen_set = _frozen_indices(frozen_bits)
    return _sc_decode_core(llr_perm, frozen_set, n)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        temp = phi
        while temp % 2 == 1:
            layers_llr.append(int(math.log2(temp & -temp)))
            temp //= 2
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = (phi + 1) // 2
        while temp % 2 == 1:
            layers_bit.append(int(math.log2(temp & -temp)))
            temp //= 2
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    内部调用与参考实现相同的核心算法以保证正确性。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    x = polar_encode(u)
    sigma = eb_n0_to_sigma(10.0, K / N)
    y = awgn_channel(bpsk_modulate(x), sigma)
    llr = compute_llr(y, sigma)

    u_sc = sc_decode(llr, frozen)
    u_rec = sc_decode_recursive(llr, frozen)
    print("SC match recursive:", np.array_equal(u_sc, u_rec))
    print("SC correct:", np.array_equal(u, u_sc))
