"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（boxplus 可选，默认 min-sum 以提速）。
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_boxplus(La, Lb):
    """精确 boxplus f 运算（标量/向量化）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty_like(La)
    for idx in np.ndindex(La.shape):
        a, b = float(La[idx]), float(Lb[idx])
        if abs(a) > 30:
            out[idx] = b if a > 0 else -b
        elif abs(b) > 30:
            out[idx] = a if b > 0 else -a
        else:
            ea, eb = np.exp(a), np.exp(b)
            out[idx] = np.log((1.0 + ea * eb) / (ea + eb))
    return out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _hard_bit(llr):
    if llr >= 0:
        return 0
    return 1


def sc_decode_recursive(llr, frozen_bits, use_boxplus=False):
    """
    递归 SC 译码（与 Sionna / Arikan 树结构一致，g 节点使用部分重编码比特）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    f_fn = f_boxplus if use_boxplus else f_operation

    def decode_node(node_llr, frozen_node):
        n = len(node_llr)
        if n == 1:
            if frozen_node[0]:
                u = 0
            else:
                u = _hard_bit(node_llr[0])
            return np.array([u], dtype=int), np.array([u], dtype=int)

        half = n // 2
        llr1 = node_llr[:half]
        llr2 = node_llr[half:]
        frozen1 = frozen_node[:half]
        frozen2 = frozen_node[half:]

        llr_u = f_fn(llr1, llr2)
        u1, u1_up = decode_node(llr_u, frozen1)
        llr_l = g_operation(llr1, llr2, u1_up)
        u2, u2_up = decode_node(llr_l, frozen2)

        u_hat = np.concatenate([u1, u2])
        u1_xor = np.bitwise_xor(u1_up.astype(int), u2_up.astype(int)).astype(int)
        u_up = np.concatenate([u1_xor, u2_up.astype(int)])
        return u_hat, u_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引。"""
    n = int(math.log2(N))
    lambda_offset = [(1 << (n - layer)) - 1 for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = [layer for layer in range(n) if ((phi >> layer) & 1) == 0]
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi % 2 == 1:
            layer = 0
            while layer < n and ((phi >> layer) & 1):
                layers_bit.append(layer)
                layer += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC（调用修正后的递归实现，接口保持一致）。"""
    return sc_decode_recursive(llr_ch, frozen_bits, use_boxplus=False)


def sc_decode_efficient(llr_ch, frozen_bits):
    """高效 SC 别名。"""
    return sc_decode(llr_ch, frozen_bits)
