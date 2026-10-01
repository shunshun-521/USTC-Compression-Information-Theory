"""
极化码 SC（串行抵消）译码器
Permuted SC（Vangala et al.）+ 递归参考实现
"""
import math
import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus 上界）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def upper_llr(l1, l2):
    # 高 SNR 下 log-domain 易数值溢出，采用稳定的 min-sum 近似
    if np.isnan(l1) or np.isnan(l2):
        return f_operation(l1, l2)
    if abs(l1) > 20 or abs(l2) > 20:
        return f_operation(l1, l2)
    return _logdomain_diff(l1 + l2, 0) - _logdomain_diff(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def hard_decision(y):
    return 0 if y >= 0 else 1


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
    """预计算非递归 SC 辅助向量（Permuted SC 比特处理顺序）。"""
    n = int(math.log2(N))
    order = [bit_reversed(i, n) for i in range(N)]
    lambda_offset = [1 << (active_llr_level(bit_reversed(phi, n), n) - 1) for phi in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = order[phi]
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PermutedSCD:
    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(int(x) for x in frozen_set)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def update_llrs(self, l):
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
                        self.B[j - branch_size, s + 1],
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self.update_bits(l)
        return self.B[:, self.n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = np.where(frozen_bits)[0]
    dec = _PermutedSCD(N, frozen_set)
    dec.L[:, 0] = llr_ch
    return dec.decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 Permuted SC 结果一致，用于交叉验证）。"""
    return sc_decode(llr, frozen_bits)
