"""
极化码 SC（串行抵消）译码器
Vangala 置换 SC（非递归）+ 递归参考实现
"""
import numpy as np
from encoder import bit_reversed_index


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算：默认 min-sum（数值稳定）；小 LLR 时可选对数域"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf if l1 > 0 and l2 > 0 else (-np.inf if l1 < 0 and l2 < 0 else 0.0)
    return f_operation(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算（LLR 域）"""
    b = int(b)
    return l1 + l2 if b == 0 else l1 - l2


def f_operation(La, Lb):
    """min-sum 近似 f（向量化，供 SCL/BP 使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _frozen_set(frozen_bits, N):
    fb = np.asarray(frozen_bits).astype(bool)
    if fb.shape[0] != N:
        raise ValueError("frozen_bits length mismatch")
    return set(np.nonzero(fb)[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（min-sum，自然序，仅作交叉验证）"""
    llr = np.asarray(llr, dtype=np.float64)
    N = llr.shape[0]
    frozen = np.asarray(frozen_bits).astype(bool)

    def dec(node_llr, off):
        n = len(node_llr)
        if n == 1:
            if frozen[off]:
                return np.array([0], dtype=int)
            return np.array([0 if node_llr[0] >= 0 else 1], dtype=int)
        h = n // 2
        ul = dec(f_operation(node_llr[:h], node_llr[h:]), off)
        ur = dec(g_operation(node_llr[:h], node_llr[h:], ul), off + h)
        return np.concatenate([ul, ur])

    return dec(llr, 0)


class _SCDCore:
    """Vangala 置换 SC 内核（与 polar-codes SCD 一致）"""

    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = frozen_set
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
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

    def decode(self, llr_ch):
        self.L[:, 0] = llr_ch
        self.B.fill(np.nan)
        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def precompute_sc_indices(N):
    """保留接口：返回 Vangala 译码相位顺序"""
    n = int(np.log2(N))
    order = [bit_reversed_index(i, n) for i in range(N)]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [list(range(n)) if i == 0 else [] for i in range(N)], order


def sc_decode(llr_ch, frozen_bits):
    """非递归 Vangala 置换 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = llr_ch.shape[0]
    frozen_set = _frozen_set(frozen_bits, N)
    return _SCDCore(N, frozen_set).decode(llr_ch)
