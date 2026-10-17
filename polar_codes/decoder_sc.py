"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归（调用同一核心）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log 域 f 运算（check node）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算（variable node）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u) * La


def _frozen_mask(frozen_bits):
    """统一冻结位掩码：True 表示冻结为 0"""
    frozen = np.asarray(frozen_bits)
    if frozen.dtype != bool:
        frozen = frozen != 0
    return frozen


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（层索引列表）"""
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        l = 0
        while l < n and not (phi & (1 << l)):
            layers.append(l)
            l += 1
        llr_layer_vec.append(layers)
        bit_layer_vec.append([l for l in range(n) if (phi >> l) & 1])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码（L=1 的 SCL，与列表译码树结构一致）"""
    from decoder_scl import SCLDecoder

    N = len(llr_ch)
    u_hat, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr_ch)
    return u_hat
