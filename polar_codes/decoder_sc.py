"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
from sc_backend import sc_decode_core


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：(1-2u)*La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算接口（兼容实验脚本）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n, 0, -1)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _frozen_to_info(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主入口"""
    info = _frozen_to_info(frozen_bits)
    return sc_decode_core(np.asarray(llr_ch, dtype=np.float64), info, frozen_bit=0)


def sc_decode_recursive(llr, frozen_bits):
    """与 sc_decode 等价（保留 API）"""
    return sc_decode(llr, frozen_bits)
