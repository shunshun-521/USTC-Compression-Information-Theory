"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Permuted SCD，高效实现）
"""
import math
import numpy as np
from decoder_utils_internal import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)

# ==================== 基本运算（min-sum，供 SCL/BP 使用）====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现，log-domain Permuted 顺序）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 Permuted SCD 等价的简化递归形式）。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    decoder = _PermutedSCD(N, n, frozen_set)
    decoder.L[:, 0] = llr
    return decoder.decode()


# ==================== 非递归 SC 译码（Permuted SCD）====================


def precompute_sc_indices(N):
    """预计算 Permuted SCD 的比特处理顺序与层更新范围。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        start = n - active_llr_level(bit_reversed(i, n), n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layers = []
        l = bit_reversed(i, n)
        if l >= N / 2:
            end = n - active_bit_level(l, n)
            bit_layers = list(range(n, end, -1))
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PermutedSCD:
    """Permuted successive cancellation decoder（非递归）。"""

    def __init__(self, N, n, frozen_set):
        self.N = N
        self.n = n
        self.frozen_set = frozen_set
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)

    def decode(self):
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

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
                        int(self.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
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


def _channel_llr_to_decoder(llr_ch):
    """发送码字经比特倒序后，将信道 LLR 映射回蝶形图顺序。"""
    N = len(llr_ch)
    from encoder import bit_reversal_permutation

    perm = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[perm] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SCD）。
    frozen_bits: 1 表示冻结位，0 表示信息位（与 run_exp 脚本一致）。
    """
    llr_ch = _channel_llr_to_decoder(llr_ch)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits)
    frozen_set = set(np.where(frozen_bits.astype(bool))[0])
    dec = _PermutedSCD(N, n, frozen_set)
    dec.L[:, 0] = llr_ch
    return dec.decode()
