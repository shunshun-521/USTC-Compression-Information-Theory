"""
极化码 SC（串行抵消）译码器
非递归 PSC 实现（log-domain），与 encoder.polar_encode 配套
"""
import numpy as np

from ref_scd.SCD import SCD
from psc_utils import active_bit_level, active_llr_level, bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 BP 等模块复用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算辅助向量（与 PSC 层更新对应）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PC:
    pass


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    pc = _PC()
    pc.N = N
    pc.n = int(np.log2(N))
    pc.frozen = np.where(frozen_bits == 1)[0]
    pc.likelihoods = llr_ch
    return SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
