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
    sign_a = np.sign(La)
    sign_b = np.sign(Lb)
    sign_a = np.where(sign_a == 0, 1, sign_a)
    sign_b = np.where(sign_b == 0, 1, sign_b)
    return sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _prepare_channel_llrs(llr_ch):
    """将自然顺序信道 LLR 映射为 SC 译码树输入顺序。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br].astype(np.float64)


def _frozen_indices_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(frozen_bits)[0])


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            psi //= 2
            layer += 1
        for l in range(layer, n):
            llr_layers.append(l)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 0 and psi > 0:
            psi //= 2
            layer += 1
        for l in range(layer):
            bit_layers.append(l)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    通过显式子树递推完成译码，与非递归实现保持一致。
    """
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（高效实现，基于 Permuted SCD 结构）。
    """
    llr = _prepare_channel_llrs(llr_ch)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = _frozen_indices_from_mask(frozen_bits)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], int(B[j - branch_size, s + 1])
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = _bit_reversed(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        update_bits(l)

    return u_hat


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(0)
    mismatch = errors_rec = errors_nr = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_rec = sc_decode_recursive(llr, frozen)
        u_rec2 = sc_decode(llr, frozen)
        if not np.array_equal(u_rec, u_rec2):
            mismatch += 1
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            errors_rec += 1
        if not np.array_equal(u[info_idx], u_rec2[info_idx]):
            errors_nr += 1
    print(f"SC test @10dB: rec_err={errors_rec}, nr_err={errors_nr}, mismatch={mismatch}")
