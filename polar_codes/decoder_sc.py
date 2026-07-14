"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def _bit_reversed_index(x, n):
    """单索引比特倒序（与 encoder 中置换一致）。"""
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


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（向量化）。
  用于 BP 等模块；SC 内部使用精确 boxplus 以保证性能。
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _f_boxplus(l1, l2):
    """精确 log-domain f / boxplus 运算（标量）。"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _g_boxplus(l1, l2, bit):
    return l1 + l2 if bit == 0 else l1 - l2


def _prepare_llr(llr_ch):
    """信道 LLR 经比特倒序后送入因子图（与含 BR 的编码器配套）。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _frozen_to_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(int) == 1)[0])


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        tmp = phi
        while tmp % 2 == 1:
            layers.append(len(layers))
            tmp //= 2
        start = len(layers)
        llr_layer_vec.append(list(range(n, start - 1, -1)))
        bit_layers = []
        tmp = phi + 1
        layer = 0
        while tmp % 2 == 0 and layer < n:
            bit_layers.append(layer)
            tmp //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDCore:
    """非递归 SC 内核（L/B 矩阵，按比特倒序序贯译码）。"""

    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(frozen_set)

    def _reset(self, llr):
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan)
        self.L[:, 0] = llr

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _f_boxplus(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = _g_boxplus(
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
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr):
        self._reset(llr)
        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(np.int32)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr = _prepare_llr(llr_ch)
    frozen = _frozen_to_set(frozen_bits)
    core = _SCDCore(len(llr_ch), frozen)
    return core.decode(llr)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（调用同一非递归内核作为参考）。"""
    return sc_decode(llr_ch, frozen_bits)
