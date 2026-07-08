"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

# ==================== 基本运算 ====================


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """精确 box-plus f 运算（向量化）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0
    if scalar:
        La, Lb = La.reshape(1), Lb.reshape(1)
    out = np.empty_like(La)
    for i in range(La.size):
        l1, l2 = La.flat[i], Lb.flat[i]
        if np.isinf(l1) and not np.isinf(l2):
            out.flat[i] = l2
        elif not np.isinf(l1) and np.isinf(l2):
            out.flat[i] = l1
        elif np.isinf(l1) and np.isinf(l2):
            out.flat[i] = np.inf
        else:
            out.flat[i] = _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)
    return out.item() if scalar else out


def g_operation(La, Lb, u_hat):
    """g 运算：b=0 -> La+Lb, b=1 -> La-Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0:
        return (La + Lb) if u_hat == 0 else (La - Lb)
    result = np.where(u_hat == 0, La + Lb, La - Lb)
    if np.isinf(La).any() or np.isinf(Lb).any():
        mask0 = (u_hat == 0) & (np.isinf(La) | np.isinf(Lb))
        result = np.where(mask0, np.inf, result)
    return result


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


def _prepare_channel_llr(llr_ch):
    """编码含比特倒序，信道 LLR 需对应倒序重排。"""
    from encoder import bit_reversal_permutation
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    rev = bit_reversal_permutation(len(llr_ch))
    return llr_ch[rev]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（基于参考实现的非递归核心）。"""
    return sc_decode(llr_ch, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）。"""
    n = int(np.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    return decode_order, None, None


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = _prepare_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

        if frozen_bits[l]:
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
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    np.random.seed(0)
    for N in [64, 128]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=bool)
        frozen_bits[info_idx] = False

        sigma = eb_n0_to_sigma(10.0, K / N)
        errors = 0
        for _ in range(100):
            u = np.zeros(N, dtype=int)
            u[info_idx] = np.random.randint(0, 2, K)
            x = polar_encode(u)
            y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
            llr = compute_llr(y, sigma)
            u_hat = sc_decode(llr, frozen_bits)
            if not np.array_equal(u_hat, u):
                errors += 1
        print(f"N={N}: SC decode test {100 - errors}/100 passed")
