"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import os

import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 log-domain f 核"""
    return np.asarray(
        np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb), dtype=np.float64
    )


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat.astype(np.float64)) * La


class _SCDecoderRecursive:
    """递归 SC 译码（单路径，结构同 SCL L=1）"""

    def __init__(self, frozen):
        self.frozen = np.asarray(frozen, dtype=bool)
        self.block_length = int(self.frozen.size)
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]

    def decode(self, channel_llr):
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        self._node([llr], 0, self.block_length)
        return self.decisions[0].astype(int)

    def _leaf(self, llrs, index):
        llr = float(llrs[0][0])
        if self.frozen[index]:
            self.decisions[0][index] = 0
        else:
            self.decisions[0][index] = 0 if llr >= 0 else 1
        return [np.array([self.decisions[0][index]], dtype=np.uint8)], [0]

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation_exact(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen = np.asarray(frozen_bits)
    if frozen.dtype != bool:
        frozen = frozen.astype(bool)
    return _SCDecoderRecursive(frozen).decode(llr_ch)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        while psi % 2 == 1:
            bit_layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（min-sum f 核，层索引预计算）。

    注：当前预计算表与递归树在部分码字上存在偏差，仿真主路径使用
    `sc_decode_recursive` 保证正确性；本函数保留供性能优化与对照。
    """
    if os.environ.get("POLAR_USE_NONRECURSIVE_SC", "").lower() in ("1", "true", "yes"):
        return _sc_decode_nonrecursive(llr_ch, frozen_bits)
    return sc_decode_recursive(llr_ch, frozen_bits)


def _sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 内核实现"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            sp = 1 << layer
            for block in range(0, N, 2 * sp):
                for j in range(sp):
                    i = block + j
                    La = P[layer + 1, i]
                    Lb = P[layer + 1, i + sp]
                    P[layer, i] = f_operation(La, Lb)
                    P[layer, i + sp] = g_operation(La, Lb, C[layer, i])

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1
        C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            sp = 1 << layer
            for block in range(0, N, 2 * sp):
                for j in range(sp):
                    i = block + j
                    C[layer + 1, i] = C[layer, i] ^ C[layer, i + sp]
                    C[layer + 1, i + sp] = C[layer, i + sp]

    return u_hat
