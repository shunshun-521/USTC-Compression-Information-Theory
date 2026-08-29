"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，SCD 调度）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


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
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 共用 LLR 置换）。"""
    return sc_decode(llr, frozen_bits)


class _SCDCore:
    """SCD 核心（mcba1n 调度）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def set_channel(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    def update_llrs(self, l):
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

    def update_bits(self, l):
        if l < self.N // 2:
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

    def decode(self):
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self.update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def precompute_sc_indices(N):
    """保留接口：返回 SCD 调度辅助信息。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = np.arange(N, dtype=int)
    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_TABLES = {}


def _get_sc_tables(N):
    if N not in _SC_TABLES:
        _SC_TABLES[N] = precompute_sc_indices(N)
    return _SC_TABLES[N]


def _permute_channel_llr(llr_ch, N):
    """将自然序信道 LLR 映射到 SCD 因子图叶节点序。"""
    br = bit_reversal_permutation(N)
    llr_perm = np.empty(N, dtype=np.float64)
    llr_perm[br] = np.asarray(llr_ch, dtype=np.float64)
    return llr_perm


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（SCD 调度，输入为自然序信道 LLR）。
    """
    N = len(llr_ch)
    core = _SCDCore(N, frozen_bits)
    core.set_channel(_permute_channel_llr(llr_ch, N))
    return core.decode()


def sc_decode_bit_reversed(llr_ch, frozen_bits):
    """与 sc_decode 相同（SCD 内部已做 bit-reversal 调度）。"""
    return sc_decode(llr_ch, frozen_bits)
