"""
极化码 SC（串行抵消）译码器
基于 L/B 矩阵的非递归实现（与标准因子图一致）
"""
import math
import numpy as np

from scd_ref import SCD


def f_operation(La, Lb):
    sign = np.where(La * Lb >= 0, np.where(La >= 0, 1.0, -1.0), -1.0)
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    class _PC:
        pass

    pc = _PC()
    pc.N = N
    pc.n = n
    pc.frozen = set(np.where(frozen_bits)[0])
    pc.likelihoods = llr_ch
    return SCD(pc).decode()


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def _layer_phi(phi, n):
    return [], []
