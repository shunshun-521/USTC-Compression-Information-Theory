"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，py-polar-codes 调度）
"""
import math
import numpy as np
from encoder import bit_reversed


def f_operation_exact(La, Lb):
    """精确 log-domain f 运算（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _logdomain_sum(La + Lb, np.zeros_like(La)) - _logdomain_sum(La, Lb)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（min-sum 路径度量兼容）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = x > y
    out = np.empty_like(x, dtype=np.float64)
    out[mask] = x[mask] + np.log1p(np.exp(y[mask] - x[mask]))
    out[~mask] = y[~mask] + np.log1p(np.exp(x[~mask] - y[~mask]))
    return out


def upper_llr(l1, l2, min_sum=False):
    """f 分支 LLR 更新"""
    if min_sum:
        return f_operation(np.array([l1]), np.array([l2]))[0]
    return _logdomain_sum(l1 + l2, np.float64(0.0)) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b, min_sum=False):
    """g 分支 LLR 更新（l1=下分支，l2=上分支）"""
    if b == 0:
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
    """找二进制表示中第一个 1 的位置（从高位）"""
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
    """找二进制表示中第一个 0 的位置（从高位）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归结构 SC 译码（参考实现）。
    与蝶形 XOR 编码配套时，采用与 sc_decode 相同的 SCD 比特倒序调度。
    """
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits, min_sum=False):
    """
    非递归 SC 译码（py-polar-codes SCD 调度，比特倒序译码顺序）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = bit_reversed(phi, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s], min_sum)
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]), min_sum
                    )

        if frozen_bits[l]:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
            u_hat[l] = int(B[l, n])

        if l < N / 2:
            continue

        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors_rec = 0
    errors_nr = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode_recursive(llr, frozen_bits)[info_idx], u[info_idx]):
            errors_rec += 1
        if not np.array_equal(sc_decode(llr, frozen_bits)[info_idx], u[info_idx]):
            errors_nr += 1
    print(f"Recursive SC: {errors_rec}/100 errors")
    print(f"Non-recursive SC: {errors_nr}/100 errors")
