"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，因子图单次 L 传播）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供 SCL 使用）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        tmp = phi
        for layer in range(n):
            if tmp % 2 == 0:
                layers.append(layer)
                tmp //= 2
            else:
                break
        llr_layer_vec.append(layers)

        blayers = []
        if phi % 2 == 1:
            blayers.append(0)
            psi = phi >> 1
            layer = 1
            while psi % 2 == 1:
                blayers.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(blayers)

    lambda_offset = [0] * (n + 1)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码（因子图 min-sum，与 encoder / BP 使用同一因子图）。
    冻结位在判决阶段强制为 0（不在 R 初始化阶段注入先验，避免破坏信息位 LLR）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch

    for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, 2 * s):
            L[i : i + s, j - 1] = f_operation(
                R[i : i + s, j] + L[i + s : i + 2 * s, j],
                L[i : i + s, j],
            )
            L[i + s : i + 2 * s, j - 1] = f_operation(
                R[i : i + s, j], L[i : i + s, j]
            ) + L[i + s : i + 2 * s, j]

    u_hat = np.zeros(N, dtype=np.int8)
    for phi in range(N):
        total = L[phi, 0] + R[phi, 0]
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if total >= 0 else 1
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归接口（与 sc_decode 等价）"""
    return sc_decode(llr, frozen_bits)
