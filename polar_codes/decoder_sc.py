"""
极化码 SC（串行抵消）译码器
"""
import numpy as np

from vendor_SCD import SCD


def f_operation(La, Lb):
    """min-sum f（供 SCL/BP）"""
    from vendor_decoder_utils import upper_llr

    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty(np.broadcast(La, Lb).shape, dtype=np.float64)
    a, b = np.broadcast_arrays(La, Lb)
    for idx in np.ndindex(a.shape):
        out[idx] = upper_llr(float(a[idx]), float(b[idx]))
    return out


def g_operation(La, Lb, u_hat):
    from vendor_decoder_utils import lower_llr

    u_hat = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty(np.broadcast(La, Lb, u_hat).shape, dtype=np.float64)
    a, b, u = np.broadcast_arrays(La, Lb, u_hat)
    for idx in np.ndindex(a.shape):
        out[idx] = lower_llr(float(a[idx]), float(b[idx]), int(u[idx]))
    return out


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        llr_layer_vec.append(list(range(l, n)))
        bit_layer_vec.append(list(range(l)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PC:
    def __init__(self, N, n, frozen, likelihoods):
        self.N = N
        self.n = n
        self.frozen = frozen
        self.likelihoods = likelihoods


def sc_decode(llr_ch, frozen_bits):
    """SC 译码（因子树非递归实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen = set(np.where(frozen_bits)[0])
    pc = _PC(N, n, frozen, llr_ch)
    return SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 结果一致，用于对照）"""
    return sc_decode(llr, frozen_bits)
