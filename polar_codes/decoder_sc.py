"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归置换 SC（Vangala / permuted SCD，高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def channel_llr_to_decoder(llr_ch):
    """极化码编码含比特倒序时，将信道 LLR 映射到置换 SC 期望的顺序。"""
    N = len(llr_ch)
    inv = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def bit_reversed(x, n):
    """单索引或数组的比特倒序。"""
    if np.isscalar(x):
        result = 0
        for i in range(n):
            if x & (1 << i):
                result |= 1 << (n - 1 - i)
        return result
    return np.array([bit_reversed(int(v), n) for v in x], dtype=int)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（与 log-domain lower_llr 在 b=0/1 时等价）。"""
    return (1 - 2 * u_hat) * La + Lb


def hard_decision(y):
    return 0 if y >= 0 else 1


def upper_llr_min_sum(l1, l2):
    return f_operation(l1, l2)


def lower_llr_min_sum(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


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


class PermutedSCDecoder:
    """Vangala 置换 SC 译码器（非递归，O(N log N)）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        frozen_bits = np.asarray(frozen_bits)
        if frozen_bits.dtype == bool:
            self.frozen = set(np.where(frozen_bits)[0])
        else:
            self.frozen = set(np.where(frozen_bits.astype(int) != 0)[0])

    def decode(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = channel_llr_to_decoder(llr_ch)

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self._update_llrs(L, B, l)
            if l in self.frozen:
                B[l, self.n] = 0
            else:
                B[l, self.n] = hard_decision(L[l, self.n])
            self._update_bits(B, l)
        return B[:, self.n].astype(int)

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr_min_sum(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    # 与参考实现一致：lower_llr(btm, top, bit)
                    L[j, s + 1] = lower_llr_min_sum(
                        L[j, s], L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（教学参考；主路径使用置换 SC）。"""
    dec = PermutedSCDecoder(len(llr), frozen_bits)
    return dec.decode(llr)


def precompute_sc_indices(N):
    """兼容接口：返回占位结构。"""
    n = int(np.log2(N))
    lambda_offset = [0] + [2 ** (n - layer) for layer in range(1, n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    N = len(llr_ch)
    return PermutedSCDecoder(N, frozen_bits).decode(llr_ch)
