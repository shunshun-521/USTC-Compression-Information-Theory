"""
极化码 SC（串行抵消）译码器
Permuted SCD（参考 Vangala et al. / mcba1n 实现）
"""
import math
import numpy as np


def bit_reversed(x, n):
    """比特倒序（标量或数组）。"""
    if np.isscalar(x):
        result = 0
        for i in range(n):
            if x & (1 << i):
                result |= 1 << (n - 1 - i)
        return result
    return np.array([bit_reversed(int(v), n) for v in x], dtype=np.int64)


def f_operation(La, Lb):
    """min-sum 近似 f（供参考/测试）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（与 log-domain lower_llr 一致）。"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def precompute_sc_indices(N):
    """预计算层索引（与非递归对照实现兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        llr_layer_vec.append(list(range(l, n)))
        if phi == N - 1:
            bit_layers = list(range(n))
        else:
            b = 0
            while b < n and (((phi + 1) >> b) & 1):
                b += 1
            bit_layers = list(range(b))
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _map_channel_llr(llr_ch):
    """将含比特倒序编码的信道 LLR 映射到 Permuted SCD 期望的顺序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    rev = bit_reversed(np.arange(N), n)
    inv = np.empty(N, dtype=np.int64)
    inv[rev] = np.arange(N)
    return llr_ch[inv]


def _permuted_scd(llr_ch, frozen_bits):
    llr_ch = _map_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def update_bits(l):
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for l in [bit_reversed(i, n) for i in range(N)]:
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        update_bits(l)

    return B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（包装 Permuted SCD，保持接口一致）。"""
    return _permuted_scd(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口。"""
    return _permuted_scd(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """别名：当前非递归实现即 Permuted SCD。"""
    return _permuted_scd(llr_ch, frozen_bits)
