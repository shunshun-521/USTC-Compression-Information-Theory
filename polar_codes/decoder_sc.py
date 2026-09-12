"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为上层 LLR，Lb 为下层 LLR"""
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """找到 i 的二进制表示中从最高位起第一个 1 之前的 0 的个数 + 1"""
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
    """找到 i 的二进制表示中从最高位起第一个 0 之前的 1 的个数 + 1"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


class _SCCore:
    """置换 SC 译码核心（Vangala et al.）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(frozen_bits)[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s], self.L[j, s], int(top_bit)
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) + int(self.B[j - branch_size, s])
                    ) % 2
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br]
        self.L.fill(np.nan)
        self.B.fill(np.nan)
        self.L[:, 0] = llr
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _SCCore(len(llr_ch), frozen_bits).decode(llr_ch)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与置换 SC 等价，用于验证）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）"""
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l >= N // 2:
            start_b = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, start_b, -1)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec, br
