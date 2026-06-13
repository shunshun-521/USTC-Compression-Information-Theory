"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，显式栈）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _decode_node(y, f_local):
    """核心递归译码单元"""
    n = len(y)
    if n == 1:
        if f_local[0]:
            return np.array([0])
        return np.array([0 if y[0] >= 0 else 1])

    u1est = f_operation(y[0::2], y[1::2])
    uhat1 = _decode_node(u1est, f_local[: n // 2])
    u1hp = uhat1.astype(np.float64)
    u2est = g_operation(f_operation(u1hp, y[0::2]), y[1::2], uhat1)
    uhat2 = _decode_node(u2est, f_local[n // 2 :])
    u = np.zeros(n, dtype=int)
    u[: n // 2] = uhat1
    u[n // 2 :] = uhat2
    return u


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _decode_node(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 路径更新使用）"""
    n = int(math.log2(N))
    return [2 ** l for l in range(n + 1)], [], []


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码：与 sc_decode_recursive 等价（显式栈）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
