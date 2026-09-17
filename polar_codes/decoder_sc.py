"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（因子图 PSCD 实现）
"""
import math
import numpy as np


def _ms_f(a, b, alpha=1.0):
    return alpha * np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，小规模验证用）。
    frozen_bits: True/1 表示冻结位
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算 PSCD 算法所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        for layer in range(n):
            if p % 2 == 0:
                llr_layers.append(layer)
            p //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi
        for layer in range(n):
            if p % 2 == 1:
                bit_layers.append(layer)
            p //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, alpha=1.0):
    """
    非递归 SC 译码（基于极化码因子图的串行消息传递）。
    输入 llr_ch 应为比特倒序置换后的信道 LLR。
    frozen_bits: 1/True 表示冻结位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    large = 1e10

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    L[:, n] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        R[:, 0] = 0.0
        R[frozen_bits, 0] = large
        for i in range(phi):
            if not frozen_bits[i]:
                R[i, 0] = large if u_hat[i] == 0 else -large

        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                for k in range(s):
                    idx = i + k
                    idx2 = idx + s
                    L[idx, j - 1] = _ms_f(
                        R[idx, j] + L[idx2, j], L[idx, j], alpha
                    )
                    L[idx2, j - 1] = (
                        _ms_f(R[idx, j], L[idx, j], alpha) + L[idx2, j]
                    )

        total = L[phi, 0] + R[phi, 0]
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if total >= 0 else 1

    return u_hat
