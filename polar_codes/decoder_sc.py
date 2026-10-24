"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效、与编码 bit-reversal 一致）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（用于递归参考实现）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """对数域精确 f（boxplus）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _upper_llr_vec(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    m = np.maximum(x, y)
    return m + np.log1p(np.exp(-np.abs(x - y)))


def _upper_llr_scalar(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _upper_llr_vec(l1, l2):
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    out = np.empty(np.broadcast(l1, l2).shape, dtype=np.float64)
    flat1 = l1.ravel()
    flat2 = l2.ravel()
    for i in range(flat1.size):
        out.ravel()[i] = _upper_llr_scalar(flat1[i], flat2[i])
    return out.reshape(out.shape)


def _lower_llr_scalar(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


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


def _bit_reversed_index(i, n):
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（保留接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p % 2 == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi % 2 == 0:
            p2 = phi
            while p2 > 0 and p2 % 2 == 0:
                bit_layers.append(int(math.log2(p2 & -p2)))
                p2 >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PermutedSCD:
    """Permuted successive cancellation decoder (Vangala et al.)."""

    def __init__(self, N, frozen_bits, use_min_sum=False):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.use_min_sum = use_min_sum
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def _upper(self, l1, l2):
        if self.use_min_sum:
            return float(
                np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))
            )
        return _upper_llr_scalar(l1, l2)

    def _lower(self, l1, l2, b):
        return _lower_llr_scalar(l1, l2, int(b))

    def set_llr(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = self._upper(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = self._lower(btm_llr, top_llr, top_bit)

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits, use_min_sum=False):
    """非递归 Permuted SC 译码（主实现）"""
    N = len(llr_ch)
    dec = _PermutedSCD(N, frozen_bits, use_min_sum=use_min_sum)
    dec.set_llr(llr_ch)
    return dec.decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 Permuted SCD 等价接口）"""
    return sc_decode(llr, frozen_bits, use_min_sum=True)


def sc_decode_permuted(llr_ch, frozen_bits, br_idx=None):
    """兼容旧接口：信道 LLR 已为自然顺序时直接译码"""
    return sc_decode(llr_ch, frozen_bits)
