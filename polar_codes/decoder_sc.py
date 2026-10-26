"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    u_hat 可为标量或与 La 同形的部分和比特向量（极化码 beta）。
    """
    return (1 - 2 * u_hat) * La + Lb


def _prepare_channel_llr(llr_ch):
    """将信道 LLR 映射到极化因子图（与 polar_encode 的比特倒序一致）。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（因子图上的 beta 递归，与 SCL L=1 等价）。
    输入 LLR 须为因子图顺序（已对信道 LLR 做比特倒序）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def leaf(index, llr_scalar):
        if frozen_bits[index]:
            u_hat[index] = 0
        else:
            u_hat[index] = 0 if llr_scalar >= 0 else 1
        return np.array([u_hat[index]], dtype=int)

    def node(llr_node, base, length):
        if length == 1:
            return leaf(base, float(llr_node[0]))

        half = length // 2
        beta_u = node(f_operation(llr_node[:half], llr_node[half:]), base, half)
        beta_l = node(
            g_operation(llr_node[:half], llr_node[half:], beta_u), base + half, half
        )
        return np.concatenate([beta_u ^ beta_l, beta_l])

    node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = phi
        layers_llr = []
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi2 = phi
            layer_id = 0
            while layer_id < n:
                if psi2 == 0 or psi2 % 2 == 0:
                    layers_bit.append(layer_id)
                psi2 >>= 1
                layer_id += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode_nonrecursive(llr, frozen_bits):
    """
    非递归 SC：当前委托给递归实现（索引布局与因子图 beta 一致）。
    保留接口供后续替换为全分层实现。
    """
    return sc_decode_recursive(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码入口：接收信道 LLR。"""
    llr = _prepare_channel_llr(llr_ch)
    return sc_decode_recursive(llr, frozen_bits)
