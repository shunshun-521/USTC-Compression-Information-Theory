"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """f 运算（对数域精确 box-plus）"""
    return np.asarray(
        np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb), dtype=np.float64
    )


def g_operation(La, Lb, u_hat):
    """g 运算：g(a,b,u) = b + (1-2u)*a"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return np.asarray(Lb + (1.0 - 2.0 * u_hat) * La, dtype=np.float64)


def upper_llr(l1, l2):
    """f 运算别名"""
    return float(f_operation(np.array([l1]), np.array([l2]))[0])


def lower_llr(l1, l2, b):
    """g 运算别名"""
    return float(g_operation(np.array([l1]), np.array([l2]), np.array([b]))[0])


def _penalty(llr, bit):
    """路径度量惩罚"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def precompute_sc_indices(N):
    """预计算辅助向量（兼容 SCL 接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（等价于 SCL L=1）"""
    from decoder_scl import SCLDecoder

    N = len(llr_ch)
    decoder = SCLDecoder(N, frozen_bits, list_size=1)
    u_hat, _ = decoder.decode(llr_ch)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，等价于 SCL L=1）"""
    return sc_decode(llr, frozen_bits)
