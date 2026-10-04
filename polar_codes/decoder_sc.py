"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f 核"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 核"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC：调用与 SCL 相同的树遍历（L=1）。
    """
    from decoder_scl import SCLDecoder

    frozen = np.asarray(frozen_bits, dtype=bool)
    if frozen.dtype != bool:
        frozen = frozen.astype(bool)
    dec = SCLDecoder(len(llr_ch), frozen, list_size=1, crc_length=0)
    u_hat, pm = dec.decode(llr_ch)
    return u_hat, pm


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 L=1 SCL 等价）"""
    return sc_decode(llr, frozen_bits)[0]


def precompute_sc_indices(N):
    """
    非递归 SC 预计算索引（保留接口；当前实现使用树递归 SCL L=1）。
    """
    import math

    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        for l in range(n):
            if ((phi >> l) & 1) == 0:
                layers = list(range(l, n))
                break
        if not layers:
            layers = list(range(n))
        llr_layer_vec.append(layers)
        if phi == 0:
            bit_layers = []
        else:
            t = phi
            cnt = 0
            while (t & 1) == 1:
                t >>= 1
                cnt += 1
            bit_layers = list(range(cnt))
        bit_layer_vec.append(bit_layers)
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
