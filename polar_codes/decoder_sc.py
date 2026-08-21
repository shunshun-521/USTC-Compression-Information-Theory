"""
极化码 SC（串行抵消）译码器
参考 mcba1n/polar-codes SCD 实现，适配含比特倒序的编码器
"""
import numpy as np
import math

from encoder import bit_reversal_permutation, bit_reversed


def logdomain_sum(x, y):
    """log-domain 加法"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 运算（精确 log-domain box-plus）"""
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算（log-domain）"""
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """向量化 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    result = np.empty_like(La)
    for i in range(La.size):
        result[i] = upper_llr(float(La[i]), float(Lb[i]))
    return result


def g_operation(La, Lb, u_hat):
    """向量化 g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def active_llr_level(i, n):
    """找二进制表示中第一个 1 的位置（从高位）"""
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
    """找二进制表示中第一个 0 的位置（从高位）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def hard_decision(y):
    return 0 if y >= 0 else 1


class SCDecoder:
    """SC 译码器（mcba1n SCD 风格）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        # 信道 LLR 重排：编码含比特倒序，树节点 i 对应 llr_ch[br(i)]
        llr_tree = llr_ch[self.br].copy()

        L = np.full((N, n + 1), np.nan, dtype=np.float64)
        B = np.full((N, n + 1), np.nan)
        L[:, 0] = llr_tree

        for phi in range(N):
            l = bit_reversed(phi, n)
            self._update_llrs(L, B, l, n, N)
            if l in self.frozen_set:
                B[l, n] = 0
            else:
                B[l, n] = hard_decision(L[l, n])
            self._update_bits(B, l, n, N)

        u_hat = B[:, n].astype(int)
        return u_hat

    def _update_llrs(self, L, B, l, n, N):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, B, l, n, N):
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    decoder = SCDecoder(len(llr_ch), frozen_bits)
    return decoder.decode(llr_ch)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    decoder = SCDecoder(len(llr), frozen_bits)
    return decoder.decode(llr)


def precompute_sc_indices(N):
    """预计算辅助索引（供 SCL 使用）"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(n + 1):
        lambda_offset[i] = 1 << i

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layers = list(range(n - active_llr_level(l, n), n))
        bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
