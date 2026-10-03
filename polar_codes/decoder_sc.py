"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain box-plus（与 5G / CommPy 一致）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


class _SCDecoderCore:
    """分层 SC 译码（Tal / Vangala 风格）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.llr = [np.zeros(1 << i, dtype=np.float64) for i in range(self.n + 1)]
        self.bits = [np.zeros(1 << i, dtype=np.int8) for i in range(self.n + 1)]

    def _update_llr(self, layer, phi):
        if layer == self.n:
            return
        if ((phi >> (self.n - layer - 1)) & 1) == 0:
            self._update_llr(layer + 1, phi)
            for beta in range(1 << layer):
                self.llr[layer][beta] = f_operation(
                    self.llr[layer + 1][2 * beta],
                    self.llr[layer + 1][2 * beta + 1],
                )
        else:
            self._update_llr(layer + 1, phi)
            for beta in range(1 << layer):
                self.llr[layer][beta] = g_operation(
                    self.llr[layer + 1][2 * beta],
                    self.llr[layer + 1][2 * beta + 1],
                    self.bits[layer + 1][beta],
                )

    def _update_bits(self, layer, phi):
        if layer < 0:
            return
        if phi % 2 == 1:
            self.bits[layer][phi // 2] = (
                self.bits[layer + 1][phi] ^ self.bits[layer + 1][phi - 1]
            )
            self._update_bits(layer - 1, phi // 2)
        else:
            self.bits[layer][phi // 2] = self.bits[layer + 1][phi]
            if phi % 4 == 2:
                self._update_bits(layer - 1, phi // 2)

    def decode(self, llr_ch):
        self.llr[self.n][:] = llr_ch
        u_hat = np.zeros(self.N, dtype=int)
        for phi in range(self.N):
            self._update_llr(0, phi)
            if self.frozen[phi]:
                u_hat[phi] = 0
            else:
                u_hat[phi] = 0 if self.llr[0][0] >= 0 else 1
            self.bits[self.n][phi] = u_hat[phi]
            self._update_bits(self.n - 1, phi)
        return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        t = 0
        p = phi
        while (p & 1) == 1:
            t += 1
            p >>= 1
        llr_layer_vec.append(list(range(t, n)))
        layers_bit = []
        p = phi + 1
        b = 0
        while (p >> b) & 1:
            layers_bit.append(b)
            b += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（L=1 SCL 实现，与编码器约定一致）"""
    from decoder_scl import sc_decode_via_scl

    return sc_decode_via_scl(llr_ch, frozen_bits)
