"""
极化码 SC（串行抵消）译码器
非递归实现基于 Vangala et al. SCD（与 polarcodes 算法一致）
"""
import math
import numpy as np
from _pp_scd import SCD


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 SCL/BP 使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


class _PC:
    """SCD 所需的最小极化码状态对象"""

    def __init__(self, N, llr):
        self.N = N
        self.n = int(math.log2(N))
        self.likelihoods = np.asarray(llr, dtype=np.float64)
        self.frozen = set()


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits[i]=True 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    pc = _PC(len(llr_ch), llr_ch)
    pc.frozen = set(np.where(frozen_bits)[0])
    return SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [0]
    for layer in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + (1 << (n - layer)))
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
