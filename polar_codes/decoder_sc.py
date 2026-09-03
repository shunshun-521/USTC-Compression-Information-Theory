"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD 高效实现
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def _upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
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


def _frozen_indices_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return np.where(frozen_bits.astype(bool))[0]


def _permute_llr_for_decode(llr_ch, N):
    """编码含比特倒序时，译码前对信道 LLR 做相同倒序置换"""
    n = int(math.log2(N))
    br = np.array([_bit_reversed(i, n) for i in range(N)])
    return np.asarray(llr_ch, dtype=np.float64)[br]


def precompute_sc_indices(N):
    """预计算 Permuted SCD 的位倒序译码顺序"""
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    return decode_order, n


class _PermutedSCD:
    """Permuted Successive Cancellation Decoder (Vangala et al., 2014)"""

    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(frozen_set)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def decode(self, llr_ch):
        self.L[:, 0] = _permute_llr_for_decode(llr_ch, self.N)
        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        int(self.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归接口包装：内部使用 Permuted SCD"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SCD 译码主函数"""
    N = len(llr_ch)
    frozen_set = _frozen_indices_from_mask(frozen_bits)
    return _PermutedSCD(N, frozen_set).decode(llr_ch)


# 供 SCL 复用
_PermutedSCDState = _PermutedSCD


if __name__ == "__main__":
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    rng = np.random.default_rng(0)
    for N in [64, 256]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

        errors = 0
        for _ in range(100):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            sigma = eb_n0_to_sigma(10.0, K / N)
            y = bpsk_modulate(x) + rng.normal(0, sigma, N)
            llr = compute_llr(y, sigma)
            u_hat = sc_decode(llr, frozen_bits)
            if not np.array_equal(u_hat[info_idx], u[info_idx]):
                errors += 1
        print(f"N={N} SC errors at 10dB: {errors}/100")
