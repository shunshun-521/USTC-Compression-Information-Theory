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
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(int) != 0


def sc_decode_recursive(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def recurse(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            if frozen[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        recurse(llr_left, offset)
        u_left = u_hat[offset:offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        recurse(llr_right, offset + half)

    recurse(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << d for d in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        tmp = phi
        layer = 0
        bit_layers = []
        while (tmp & 1) and layer < n:
            bit_layers.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
        llr_layer_vec.append(list(range(n - 1, layer - 1, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码（SCL L=1 等价实现）"""
    from decoder_scl import SCLDecoder

    N = len(llr_ch)
    u_hat, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr_ch)
    return u_hat
