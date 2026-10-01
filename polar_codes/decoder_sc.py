"""
极化码 SC（串行抵消）译码器
Permuted SCD（与 B_N 编码约定匹配）
"""
import numpy as np


def bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr(l1, l2):
    if abs(l1) > 80 or abs(l2) > 80:
        return float(f_operation(np.array([l1]), np.array([l2]))[0])
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u = np.asarray(u_hat)
    return (1 - 2 * u) * La + Lb


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


def _reorder_channel_llrs(llr_ch):
    """保持接口：Permuted SCD 直接使用信道顺序 LLR"""
    return np.asarray(llr_ch, dtype=np.float64)


class _PermutedSCD:
    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def decode(self, llr_ch):
        self.L[:, 0] = llr_ch
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        int(self.B[j - branch_size, s + 1]),
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


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr = _reorder_channel_llrs(llr_ch)
    N = len(llr)
    return _PermutedSCD(N, frozen_bits).decode(llr)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 参考实现（调用 Permuted SCD）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = list(range(n - active_llr_level(bit_reversed(phi, n), n), n))
        llr_layer_vec.append(layers)
        bit_layers = list(range(n, n - active_bit_level(bit_reversed(phi, n), n), -1))
        bit_layer_vec.append(bit_layers)
    return llr_layer_vec, bit_layer_vec
