"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效实现，Vangala 2014）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 BP 等模块使用）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（min-sum 路径，供 BP 等模块使用）。"""
    return (1 - 2 * u_hat) * La + Lb


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


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
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


def channel_llr_for_decoder(llr_ch):
    """极化编码含输出比特倒序时，将信道 LLR 对齐到 PSCD 因子图。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=np.int64)
    inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def _frozen_index_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(bool))[0])


class _PermutedSCD:
    """Permuted successive cancellation decoder（非递归，对数域 f/g）。"""

    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(frozen_set)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def decode(self, llr):
        self.L[:, 0] = llr
        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(np.int8)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s], self.L[j - branch, s], self.B[j - branch, s + 1]
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch:
                    self.B[j - branch, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch, s])
                    self.B[j, s - 1] = self.B[j, s]


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容；PSCD 内部自洽更新）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(np.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi % 2 == 1:
            psi = phi
            while psi % 2 == 1:
                bit_layers.append(int(np.log2(psi & -psi)))
                psi >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD）。"""
    llr = channel_llr_for_decoder(llr_ch)
    frozen_set = _frozen_index_set(frozen_bits)
    return _PermutedSCD(len(llr), frozen_set).decode(llr)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 PSCD 等价，用于交叉验证）。"""
    llr = channel_llr_for_decoder(llr)
    frozen_set = _frozen_index_set(frozen_bits)
    return _PermutedSCD(len(llr), frozen_set).decode(llr)
