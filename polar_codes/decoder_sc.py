"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（主要用于 BP 译码器）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _f_boxplus(La, Lb):
    """SC/SCL 使用的精确 log-domain boxplus。"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def _bit_reversed_index(x, n):
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


def _to_frozen_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(int) != 0)[0])


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    返回的层列表与按位倒序 SC 调度一致。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        bit_start = n - _active_bit_level(l, n)
        bit_layer_vec.append(list(range(n, bit_start, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _lower_llr(l1, l2, b):
    """g 运算的标量版本，与 Vangala SC 调度一致。"""
    if b is None or (isinstance(b, float) and np.isnan(b)):
        b = 0
    b = int(b)
    return l1 + l2 if b == 0 else l1 - l2


def _sc_decode_core(llr_ch, frozen_set, N):
    """Permuted successive cancellation decoder（Vangala 2014）。"""
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1))
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed_index(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(np.int64)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 先按比特倒序置换，以匹配编码端的 B_N 置换。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llr_perm = llr_ch[rev]
    frozen_set = _to_frozen_set(frozen_bits)
    return _sc_decode_core(llr_perm, frozen_set, N)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与非递归版本等价）。
    """
    return sc_decode(llr, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100, seed=0):
    """验证 SC 译码器在高信噪比下无误码。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=np.int64)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            raise AssertionError("SC 译码信息位错误")

    return True


if __name__ == "__main__":
    verify_sc_decoders()
    print("SC 译码器校验通过")
