"""
极化码 SC（串行抵消）译码器
Permuted SCD + log-domain f 函数
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似（供 SCL/BP 复用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(l1, l2, b):
    return g_operation(l1, l2, b)


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


class _SCDEngine:
    def __init__(self, N, n, llr):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr

    def decode(self, frozen_set):
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            bs = 1 << (s + 1)
            t = bs >> 1
            for j in range(l, self.N, bs):
                if j % bs < t:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + t, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j - t, s], self.L[j, s], int(self.B[j - t, s + 1])
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            bs = 1 << s
            t = bs >> 1
            for j in range(l, -1, -bs):
                if j % bs >= t:
                    self.B[j - t, s - 1] = int(self.B[j, s]) ^ int(self.B[j - t, s])
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """信道 LLR 对应 polar_encode 的输出比特顺序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    inv_br = np.argsort(bit_reversal_permutation(N))
    llr_v = llr_ch[inv_br]
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int))[0])
    return _SCDEngine(N, n, llr_v).decode(frozen_set)


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
