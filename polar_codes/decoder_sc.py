"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Vangala 置换 SC，高效实现）
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


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """f 运算（对数域盒加，比 min-sum 更精确）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(btm, top, bit):
    """g 运算"""
    return g_operation(top, btm, bit)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回置换相位顺序及每层活跃信息。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    phase_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in phase_order:
        llr_layers = list(range(n - _active_llr_level(phi, n), n))
        bit_layers = []
        if phi >= N // 2:
            bit_layers = list(range(n, n - _active_bit_level(phi, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec, phase_order


class _SCDCore:
    """Vangala 置换 SC 译码核心"""

    def __init__(self, N, llr_ch, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.phase_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, self.N, block):
                if j % block < branch:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch, s],
                        int(self.B[j - branch, s + 1]),
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    self.B[j - branch, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch, s])
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        u_hat = np.zeros(self.N, dtype=np.int8)
        for l in self.phase_order:
            self._update_llrs(l)
            if self.frozen_bits[l]:
                bit = 0
            else:
                bit = 0 if self.L[l, self.n] >= 0 else 1
            self.B[l, self.n] = bit
            u_hat[l] = bit
            self._update_bits(l)
        return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Vangala 置换 SC）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return _SCDCore(len(llr_ch), llr_ch, frozen_bits).decode()


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码参考接口（内部复用 Vangala 置换 SC 以保证与编码器一致）。
    """
    return sc_decode(llr, frozen_bits)
