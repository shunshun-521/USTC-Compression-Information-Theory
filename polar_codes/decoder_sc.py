"""
极化码 SC（串行抵消）译码器
非递归 SCD 实现（Vangala et al. 2014 / polar-codes）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation, polar_encode


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """La=bottom, Lb=top（与 polar-codes lower_llr 一致）"""
    if int(u_hat):
        return La - Lb
    return La + Lb


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
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
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _scd_decode_llr(llr_ch, frozen_bits):
    N = len(llr_ch)
    n = int(math.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(btm_llr, top_llr, top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    u_hat = _scd_decode_llr(llr_ch[br], frozen_bits)
    u_hat[frozen_bits] = 0
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def validate_sc_decoders(N=64, frozen_bits=None, trials=50):
    if frozen_bits is None:
        K = N // 2
        from construction import ga_construction

        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=bool)
        frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[~frozen_bits] = rng.integers(0, 2, size=int((~frozen_bits).sum()))
        x = polar_encode(u)
        llr = (1.0 - 2.0 * x) * 80.0
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_rec, u):
            raise AssertionError("SC decoder validation failed")
    return True
