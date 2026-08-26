"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

_FloatArray = np.ndarray


def f_operation(La, Lb):
    """精确 log-domain f 运算（check node）"""
    a = np.asarray(La, dtype=np.float64)
    b = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def g_operation(La, Lb, u_hat):
    """g 运算：g(a,b,u) = b + (1-2u)*a"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    b = np.asarray(Lb, dtype=np.float64)
    a = np.asarray(La, dtype=np.float64)
    return b + (1.0 - 2.0 * u_hat) * a


def _penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 commpy 树结构一致）"""
    frozen = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def leaf(llr_node, index):
        if frozen[index]:
            u_hat[index] = 0
            return np.array([0], dtype=np.int8)
        bit = 0 if llr_node[0] >= 0 else 1
        u_hat[index] = bit
        return np.array([bit], dtype=np.int8)

    def node(llr_node, base, length):
        if length == 1:
            return leaf(llr_node, base)

        half = length // 2
        upper = f_operation(llr_node[:half], llr_node[half:])
        beta_upper = node(upper, base, half)
        lower = g_operation(
            llr_node[:half], llr_node[half:], beta_upper.astype(np.float64)
        )
        beta_lower = node(lower, base + half, half)
        return np.concatenate([beta_upper ^ beta_lower, beta_lower])

    node(np.asarray(llr, dtype=np.float64), 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(1, n + 1):
        lambda_offset[i] = lambda_offset[i - 1] + (1 << (n - i))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p % 2 == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p2 = phi
        layer = 0
        while p2 % 2 == 1:
            bit_layers.append(layer)
            p2 >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：list_size=1 的 SCL 等价实现"""
    from decoder_scl import SCLDecoder

    frozen = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    decoder = SCLDecoder(N, frozen, list_size=1, crc_length=0)
    u_hat, _ = decoder.decode(llr_ch)
    return u_hat
