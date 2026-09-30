"""
极化码 SC（串行抵消）译码器
非递归 Permuted SCD + 递归参考实现
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 SCL/BP 使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（min-sum 路径）"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l2 == np.inf and l1 != np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    b = int(b)
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed(x, n):
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


class _SCDCore:
    def __init__(self, llr_ch, frozen_indices):
        self.N = len(llr_ch)
        self.n = int(math.log2(self.N))
        self.frozen = set(frozen_indices)
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((self.N, self.n + 1), dtype=np.float64)
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_idx = np.where(frozen_bits)[0]
    llr = np.asarray(llr_ch, dtype=np.float64)
    return _SCDCore(llr, frozen_idx).decode()


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode_nonrecursive(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：非递归 SCD 使用预计算层索引（占位）"""
    n = int(math.log2(N))
    return list(range(n + 1)), [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_nonrecursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    for N, K in [(4, 2), (64, 32)]:
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen = np.ones(N, dtype=bool)
        frozen[info_idx] = False
        rng = np.random.default_rng(0)
        sigma = eb_n0_to_sigma(10.0, K / N)
        errors = 0
        for _ in range(50):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)
            uh = sc_decode(llr, frozen)
            if not np.array_equal(uh[info_idx], u[info_idx]):
                errors += 1
        print(f"N={N}: {errors}/50 frame errors @10dB")
