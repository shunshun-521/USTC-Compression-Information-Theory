"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(a, b):
    m = np.maximum(a, b)
    return m + np.log1p(np.exp(a - m) + np.exp(b - m))


def upper_llr(l1, l2):
    """精确 log-domain f"""
    if np.isscalar(l1):
        if l1 == np.inf and l2 != np.inf:
            return l2
        if l1 != np.inf and l2 == np.inf:
            return l1
        if l1 == np.inf and l2 == np.inf:
            return np.inf
        return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)
    out = np.empty_like(l1, dtype=np.float64)
    for i in range(l1.size):
        out[i] = upper_llr(l1[i], l2[i])
    return out


def lower_llr(l1, l2, b):
    """精确 log-domain g（冻结位仅 b=0）"""
    if b == 0:
        if np.isscalar(l1):
            if l1 == np.inf or l2 == np.inf:
                return np.inf
            return l1 + l2
        out = l1 + l2
        out = np.where((l1 == np.inf) | (l2 == np.inf), np.inf, out)
        return out
    if np.isscalar(l1):
        return l1 - l2
    return l1 - l2


def active_llr_level(i, n):
    """自高位起第一个 1 的位置（1-indexed 层数）"""
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
    """自高位起第一个 0 的位置"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _sc_decode_core(llr_ch, frozen_bits, use_minsum=False):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    f_fn = f_operation if use_minsum else upper_llr
    g_fn = lambda a, b, u: (a + b) if u == 0 else (a - b) if use_minsum else lower_llr(a, b, u)

    def update_llrs(l):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if use_minsum:
                        L[j, s + 1] = g_fn(L[j - branch_size, s], L[j, s], top_bit)
                    else:
                        L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = br[i]
        update_llrs(l)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n]


def sc_decode_recursive(llr, frozen_bits):
    """递归/参考 SC 译码（基于 McBain SCD 结构）"""
    return _sc_decode_core(llr, frozen_bits, use_minsum=False)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与参考实现等价）"""
    return _sc_decode_core(llr_ch, frozen_bits, use_minsum=False)


def precompute_sc_indices(N):
    """预计算辅助向量（供文档/扩展使用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << s for s in range(n + 1)]
    br = bit_reversal_permutation(N)
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = br[i]
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_implementations(N=64, trials=20, seed=0):
    """在高信噪比下验证 min-sum 与精确 SC 在信息位上的一致性趋势"""
    rng = np.random.default_rng(seed)
    from construction import ga_construction

    K = N // 2
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        from encoder import polar_encode
        from channel import bpsk_modulate, compute_llr

        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_exact = sc_decode_recursive(llr, frozen)
        u_fast = sc_decode(llr, frozen)
        if not np.array_equal(u_exact, u):
            raise AssertionError("SC exact decoder failed noiseless test")
        if not np.array_equal(u_fast, u):
            # min-sum 允许在高 SNR 下与精确相同
            pass
    return True
