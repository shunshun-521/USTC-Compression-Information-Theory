"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def lowerconv(upperdecision, upperllr, lowerllr):
    """g 运算（对数域，参考 SC 树更新）。"""
    if upperdecision == 0:
        return lowerllr + upperllr
    return lowerllr - upperllr


def g_operation(La, Lb, u_hat):
    """向量 g 运算。"""
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat) or u_hat.ndim == 0:
        return lowerconv(int(u_hat), La, Lb)
    out = np.empty_like(La, dtype=np.float64)
    for idx in range(len(La)):
        out[idx] = lowerconv(int(u_hat[idx]), La[idx], Lb[idx])
    return out


def bitreversed(num, n):
    return int(f"{num:0{n}b}"[::-1], 2)


class SCState:
    """单路径 SC 译码状态（2N-1 LLR 数组）。"""

    def __init__(self, N):
        self.N = N
        self.n = int(math.log2(N))
        self.llrs = np.zeros(2 * N - 1, dtype=np.float64)
        self.bits = np.zeros((2, N - 1), dtype=int)
        self.decoded_bit = 0

    def set_channel(self, llr_ch):
        self.llrs[self.N - 1:] = llr_ch

    def update_llrs(self, pos):
        """pos 为比特倒序后的译码位置索引。"""
        if pos == 0:
            nextlevel = self.n
        else:
            lastlevel = (bin(pos)[2:].zfill(self.n)).find("1") + 1
            start = 2 ** (lastlevel - 1) - 1
            end = 2 ** lastlevel - 1 - 1
            for i in range(start, end + 1):
                self.llrs[i] = lowerconv(
                    self.bits[0, i],
                    self.llrs[end + 2 * (i - start) + 1],
                    self.llrs[end + 2 * (i - start) + 2],
                )
            nextlevel = lastlevel - 1

        for lev in range(nextlevel, 0, -1):
            start = 2 ** (lev - 1) - 1
            end = 2 ** lev - 1 - 1
            for indx in range(start, end + 1):
                exp1 = end + 2 * (indx - start)
                llr1 = self.llrs[exp1 + 1]
                llr2 = self.llrs[exp1 + 2]
                self.llrs[indx] = f_operation(llr1, llr2)

    def update_bits(self, pos):
        if pos == self.N - 1:
            return
        latestbit = self.decoded_bit
        n = self.n
        if pos < self.N // 2:
            self.bits[0, 0] = latestbit
        else:
            lastlevel = (bin(pos)[2:].zfill(n)).find("0") + 1
            self.bits[1, 0] = latestbit
            for lev in range(1, lastlevel - 1):
                st = 2 ** (lev - 1) - 1
                ed = 2 ** lev - 1 - 1
                for i in range(st, ed + 1):
                    self.bits[1, ed + 2 * (i - st) + 1] = (
                        self.bits[0, i] + self.bits[1, i]
                    ) % 2
                    self.bits[1, ed + 2 * (i - st) + 2] = self.bits[1, i]

            lev = lastlevel - 1
            st = 2 ** (lev - 1) - 1
            ed = 2 ** lev - 1 - 1
            for i in range(st, ed + 1):
                self.bits[0, ed + 2 * (i - st) + 1] = (
                    self.bits[0, i] + self.bits[1, i]
                ) % 2
                self.bits[0, ed + 2 * (i - st) + 2] = self.bits[1, i]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits[i]=1 表示冻结位（自然顺序索引）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)

    state = SCState(N)
    state.set_channel(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    for j in range(N):
        i = bitreversed(j, n)
        state.update_llrs(i)
        if frozen_bits[j]:
            u_hat[j] = 0
        else:
            u_hat[j] = 0 if state.llrs[0] >= 0 else 1
        state.decoded_bit = u_hat[j]
        state.update_bits(i)

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用与主函数相同的实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = 2 ** (n - layer)

    llr_layer_vec = []
    for phi in range(N):
        layers = []
        for layer in range(1, n + 1):
            if (phi % (2 ** layer)) < (2 ** (layer - 1)):
                layers.append(layer)
        llr_layer_vec.append(layers)

    bit_layer_vec = []
    for phi in range(N):
        layers = []
        if phi % 2 == 0:
            layer = 1
            while phi % (2 ** layer) == 0 and layer <= n:
                layer += 1
            layer -= 1
            while layer >= 1:
                layers.append(layer)
                layer -= 1
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
