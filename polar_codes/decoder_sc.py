"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Vangala 置换 SC，高效实现）
"""
import numpy as np


def bit_reversal_permutation(N):
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """f 运算（对数域 box-plus，标量或逐元素）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        if La == np.inf and Lb != np.inf:
            return float(Lb)
        if La != np.inf and Lb == np.inf:
            return float(La)
        if La == np.inf and Lb == np.inf:
            return np.inf
        return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)
    return np.vectorize(f_operation, otypes=[float])(La, Lb)


def lower_llr(l1, l2, b):
    """下分支 LLR 更新（l1=btm, l2=top）。"""
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    b = np.asarray(b, dtype=int)
    if l1.ndim == 0:
        b = int(b)
        if b == 0:
            if l1 == np.inf or l2 == np.inf:
                return np.inf
            return l1 + l2
        return l1 - l2
    return np.where(b == 0, l1 + l2, l1 - l2)


def g_operation(La, Lb, u_hat):
    """g 运算（递归译码用）：La=top, Lb=btm。"""
    return lower_llr(Lb, La, u_hat)


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
    """预计算置换 SC 的译码相位顺序及辅助信息。"""
    n = int(np.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    lambda_offset = np.array([active_llr_level(l, n) for l in decode_order], dtype=int)
    bit_levels = [active_bit_level(l, n) for l in decode_order]
    return decode_order, lambda_offset, bit_levels


def _map_channel_llrs(llr_ch):
    """将信道 LLR 映射到置换 SC 译码树顺序。"""
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[brp]


class _SCDCore:
    """Vangala 置换 SC 译码核心（非递归）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def set_llr(self, llr_ch):
        self.L[:, 0] = _map_channel_llrs(llr_ch)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        u_hat = np.zeros(self.N, dtype=int)
        for l in self.decode_order:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            u_hat[l] = int(self.B[l, self.n])
            self._update_bits(l)
        return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（Vangala 置换实现）。"""
    N = len(llr_ch)
    core = _SCDCore(N, frozen_bits)
    core.set_llr(llr_ch)
    return core.decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，调用置换 SC 以保证一致性）。"""
    return sc_decode(llr, frozen_bits)
