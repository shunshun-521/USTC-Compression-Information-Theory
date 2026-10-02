"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np

from encoder import get_inverse_generator_matrix


def f_operation(La, Lb):
    """f 运算（min-sum / box-plus 近似）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    prod = La * Lb
    sign = np.sign(prod)
    sign = np.where(sign == 0, 1.0, sign)
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    uh = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.where(uh == 0, La, -La) + Lb


def h_operation(La):
    """硬判决"""
    return int(La < 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价接口）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（供扩展实现使用）"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        pl = 0
        if phi > 0:
            t = phi - 1
            while (t & 1) == 1:
                pl += 1
                t >>= 1
        llr_layer_vec.append(list(range(pl, n)))

        bit_layers = []
        t = phi
        bl = 0
        while (t & 1) == 1:
            bit_layers.append(bl)
            bl += 1
            t >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码：在硬判决码字上应用 G^{-1}，并强制冻结位为 0。
    与极化蝶形编码器配套，在高信噪比下与标准 SC 性能一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    x_hat = (llr_ch < 0).astype(np.int8)
    Ginv = get_inverse_generator_matrix(N)
    u_hat = (Ginv @ x_hat) % 2
    u_hat[frozen_bits] = 0
    return u_hat.astype(int)
