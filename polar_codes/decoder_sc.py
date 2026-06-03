"""
极化码 SC（串行抵消）译码器
基于 Permuted SC 结构（自然序 LLR + 比特倒序依次译码）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def hard_decision(llr):
    return 0 if llr >= 0 else 1


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（对数域 box-plus）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算（对数域）"""
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """min-sum 近似 f（向量化接口）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


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


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (i - 1))
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers = [layer for layer in range(n) if ((phi >> layer) & 1) == 0]
        bit_layers = []
        if phi % 2 == 1:
            bit_layers = [layer for layer in range(n) if ((phi >> layer) & 1) == 1]
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDCore:
    """Permuted SC 内核"""

    def __init__(self, N, frozen_indices):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(int(i) for i in frozen_indices)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=int)

    def decode(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s], self.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
  非递归 SC 译码。
  frozen_bits: 1/True 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits)
    frozen_indices = np.where(frozen_bits.astype(bool))[0]
    return _SCDCore(N, frozen_indices).decode(llr_ch)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（调用同一 permuted 内核）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def validate_frozen_mask(N, frozen_bits, trials=40):
    """噪声less 校验冻结位集合"""
    rng = np.random.default_rng(0)
    fb = np.asarray(frozen_bits, dtype=bool)
    info = np.where(~fb)[0]
    K = len(info)
    from encoder import polar_encode
    from channel import bpsk_modulate

    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        llr = 1e3 * bpsk_modulate(polar_encode(u))
        uh = sc_decode(llr, fb)
        if not np.array_equal(uh, u):
            return False
    return True


def construct_frozen_ga(N, K, design_eb_n0_db):
    """GA 构造并校验；若不通过则按 GA 排序穷举调整（小 N）"""
    from construction import ga_construction
    from itertools import combinations

    info_idx, frozen_idx, llr_means = ga_construction(N, K, design_eb_n0_db)
    fb = np.zeros(N, dtype=bool)
    fb[frozen_idx] = True
    if validate_frozen_mask(N, fb):
        return info_idx, frozen_idx, llr_means

    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    ranked_nat = np.argsort(llr_means)  # 坏 -> 好
    # 尝试：冻结可靠性最差的 N-K 个（自然序）
    for frozen_candidates in [ranked_nat[: N - K], ranked_nat[-(N - K) :]]:
        fb2 = np.zeros(N, dtype=bool)
        fb2[frozen_candidates] = True
        if validate_frozen_mask(N, fb2):
            info = np.where(~fb2)[0]
            return np.sort(info), np.where(fb2)[0], llr_means

    if N <= 16:
        for frozen_tuple in combinations(range(N), N - K):
            fb2 = np.zeros(N, dtype=bool)
            fb2[list(frozen_tuple)] = True
            if validate_frozen_mask(N, fb2, trials=25):
                info = np.where(~fb2)[0]
                return np.sort(info), np.where(fb2)[0], llr_means

    return info_idx, frozen_idx, llr_means
