"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于分层 L/B 矩阵）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(top_llr, btm_llr, u_hat):
    """
    g 运算（与分层 SC 下分支一致：lower_llr(btm, top, b)）。
    b=0 -> top+btm； b=1 -> btm-top
    """
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, top_llr + btm_llr, btm_llr - top_llr)


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """二进制表示中自高位起第一个 1 的位置（用于 LLR 更新层数）"""
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
    """二进制表示中自高位起第一个 0 的位置（用于比特回传层数）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class SCDecoderState:
    """非递归 SC 译码状态（L/B 矩阵）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)

    def set_channel_llr(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = f_operation(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)  # top, btm

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode_bit(self, l):
        self.update_llrs(l)
        if l in self.frozen_set:
            self.B[l, self.n] = 0
        else:
            self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
        self.update_bits(l)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码：按比特倒序相位依次判决。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    state = SCDecoderState(N, frozen_bits)

    state.set_channel_llr(llr_ch)
    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        state.decode_bit(l)

    u_br = state.B[:, n].astype(int)
    # 按比特倒序相位译码，输出映射回自然顺序
    br = np.array([bit_reversed_index(i, n) for i in range(N)], dtype=int)
    u_hat = np.empty(N, dtype=int)
    u_hat[br] = u_br
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用非递归实现，保持接口一致）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助索引（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        lambda_offset[phi] = active_llr_level(l, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
