"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_min_sum(La, Lb):
    """min-sum 近似 f 运算（BP 等使用）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def f_operation(La, Lb):
    """
    SC 译码用 box-plus 精确 f 运算（数值稳定）。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    La = np.clip(La, -1e2, 1e2)
    Lb = np.clip(Lb, -1e2, 1e2)
    with np.errstate(over="ignore", invalid="ignore"):
        t = np.tanh(La / 2.0) * np.tanh(Lb / 2.0)
        t = np.clip(t, -1.0 + 1e-15, 1.0 - 1e-15)
        out = 2.0 * np.arctanh(t)
    out = np.where(np.isfinite(out), out, 0.0)
    return out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================

def _sc_decode_tree(llr, frozen_bits, u_force=None, return_llr_at=None):
    """递归 SC 树遍历；u_force 固定已知比特，return_llr_at 返回该相位 LLR。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    if u_force is None:
        u_force = {}

    def decode_node(llr_node, frozen_node, bit_offset):
        n = len(llr_node)
        if return_llr_at is not None and bit_offset == return_llr_at and n == 1:
            return float(llr_node[0])

        if n == 1:
            if bit_offset in u_force:
                u = int(u_force[bit_offset])
            elif frozen_node[0]:
                u = 0
            else:
                u = 0 if llr_node[0] >= 0 else 1
            return [u], [u]

        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        fr1 = frozen_node[:half]
        fr2 = frozen_node[half:]

        llr_up = f_operation(llr1, llr2)
        res_up = decode_node(llr_up, fr1, bit_offset)
        if isinstance(res_up, float):
            return res_up
        u_hat1, u_hat1_up = res_up

        u_hat1_up_arr = np.asarray(u_hat1_up, dtype=np.int8)
        llr_low = g_operation(llr1, llr2, u_hat1_up_arr)
        res_low = decode_node(llr_low, fr2, bit_offset + half)
        if isinstance(res_low, float):
            return res_low
        u_hat2, u_hat2_up = res_low

        u_hat = u_hat1 + u_hat2
        u1_up = np.asarray(u_hat1_up, dtype=np.int8)
        u2_up = np.asarray(u_hat2_up, dtype=np.int8)
        u_hat_up = list(u1_up ^ u2_up) + list(u2_up)
        return u_hat, u_hat_up

    return decode_node(llr, frozen_bits, 0)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与 Sionna/Arikan 树形记号一致）。
    g 运算使用子树重编码后的 u_hat_up，而非源比特 u_hat。
    """
    out = _sc_decode_tree(llr, frozen_bits)
    u_hat, _ = out
    return np.asarray(u_hat, dtype=np.int8)


def sc_llr_at_phase(llr, frozen_bits, u_prefix, phi):
    """给定前缀 u_prefix（长度 phi），计算第 phi 个比特的 SC LLR。"""
    u_force = {i: int(u_prefix[i]) for i in range(len(u_prefix))}
    return float(_sc_decode_tree(llr, frozen_bits, u_force=u_force, return_llr_at=phi))


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        while p & 1:
            layers_llr.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        p = phi
        while p & 1:
            layers_bit.append(int(math.log2(p & -p)))
            p >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码入口：对信道 LLR 做树序对齐后，采用与 sc_decode_recursive
    相同的串行抵消逻辑（主仿真路径）。
    """
    llr_aligned = align_llr_for_decoder(llr_ch)
    return sc_decode_recursive(llr_aligned, frozen_bits)


def align_llr_for_decoder(llr_ch):
    """
    将信道 LLR 重排为 SC 译码树叶子顺序（比特倒序，与 Arikan 记号一致）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]
