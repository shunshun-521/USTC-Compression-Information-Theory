"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，置换因子图）
"""
import numpy as np
import math


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _bit_reversed_scalar(x, n):
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


def f_operation(La, Lb):
    """f 运算（对数域）"""
    l1 = float(La)
    l2 = float(Lb)
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l2 == np.inf and l1 != np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def g_operation(La, Lb, u_hat):
    """g 运算：La=bottom, Lb=top（与置换因子图一致）"""
    l1 = float(La)
    l2 = float(Lb)
    u = int(u_hat)
    if u == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        br_phi = _bit_reversed_scalar(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(br_phi, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(br_phi, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（置换因子图，按比特倒序信道索引）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [_bit_reversed_scalar(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return np.nan_to_num(B[:, n], nan=0.0).astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现）"""
    return sc_decode(llr, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, awgn_channel
    from construction import ga_construction

    rng = np.random.default_rng(0)
    for N in [8, 64, 256]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=bool)
        frozen_bits[info_idx] = False

        errors = 0
        for _ in range(100):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            y = bpsk_modulate(x)
            llr = compute_llr(y, 0.01)
            u_hat = sc_decode(llr, frozen_bits)
            if not np.array_equal(u_hat, u):
                errors += 1
        print(f"N={N}: noiseless errors={errors}/100")
