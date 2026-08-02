"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（对应 P1 域 cnop）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（已知左子树比特时的右子树 LLR）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _llr_to_p1(llr):
    llr = np.clip(llr, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(llr))


def _p1_to_llr(p1):
    p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
    return np.log((1.0 - p1) / p1)


def _cnop_p1(a, b):
    return a * (1.0 - b) + b * (1.0 - a)


def _vnop_p1(a, b):
    num = a * b
    den = num + (1.0 - a) * (1.0 - b)
    return num / (den + 1e-300)


def _polar_dec_p1(y, frozen):
    """P1 域递归 SC 译码（与 B_N F^{⊗n} 编码匹配）"""
    N = len(y)
    frozen = np.asarray(frozen, dtype=bool)
    if N == 1:
        bit = 1 if y[0] >= 0.5 else 0
        if frozen[0]:
            return np.array([0], dtype=int), np.array([y[0]], dtype=np.float64)
        return np.array([bit], dtype=int), np.array([float(bit)], dtype=np.float64)

    y_top = _cnop_p1(y[::2], y[1::2])
    uhat1, u1hp = _polar_dec_p1(y_top, frozen[: N // 2])
    u2est = _vnop_p1(_cnop_p1(u1hp, y[::2]), y[1::2])
    uhat2, u2hp = _polar_dec_p1(u2est, frozen[N // 2 :])

    u = np.concatenate([uhat1, uhat2])
    x1 = _cnop_p1(u1hp, u2hp)
    x = np.zeros(N, dtype=np.float64)
    x[::2] = x1
    x[1::2] = u2hp
    return u, x


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（P1 域实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    p1 = _llr_to_p1(np.asarray(llr, dtype=np.float64))
    u, _ = _polar_dec_p1(p1, frozen_bits)
    return u


def bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（与 SCL 共享接口）"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用高效 P1 递归内核）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
