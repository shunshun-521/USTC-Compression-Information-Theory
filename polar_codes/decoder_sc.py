"""
极化码 SC（串行抵消）译码器
实现与 Sionna PolarSCDecoder 等价的递归树算法（boxplus + u_hat_up）
"""
import math
import numpy as np

LLR_MAX = 20.0


def f_operation(La, Lb):
    """
    检查节点 boxplus（与 Sionna _cn_op 一致，向量化）。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    x = np.clip(La, -LLR_MAX, LLR_MAX)
    y = np.clip(Lb, -LLR_MAX, LLR_MAX)
    return np.logaddexp(0.0, x + y) - np.logaddexp(x, y)


def g_operation(La, Lb, u_hat_up):
    """变量节点更新（使用 u_hat_up，而非最终 u_hat）。"""
    u_hat_up = np.asarray(u_hat_up)
    return (1.0 - 2.0 * u_hat_up) * La + Lb


def _polar_decode_sc(llr_ch, frozen_ind):
    """
    递归 SC 译码，返回 (u_hat, u_hat_up) 长度均为当前子块 n。
    frozen_ind: 长度 n，1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.int8)
    n = frozen_ind.shape[0]

    if n > 1:
        half = n // 2
        llr1 = llr_ch[..., :half]
        llr2 = llr_ch[..., half:]
        f1 = frozen_ind[:half]
        f2 = frozen_ind[half:]

        x_llr1 = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc(x_llr1, f1)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc(x_llr2, f2)

        u_hat = np.concatenate([u_hat1, u_hat2], axis=-1)
        u_hat1_up_i = (u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)).astype(
            np.float64
        )
        u_hat_up = np.concatenate([u_hat1_up_i, u_hat2_up], axis=-1)
        return u_hat, u_hat_up

    is_frozen = frozen_ind[0] == 1
    if is_frozen:
        u_leaf = np.zeros_like(llr_ch)
    else:
        u_leaf = 0.5 * (1.0 - np.sign(llr_ch))
        u_leaf = np.where(u_leaf == 0.5, 1.0, u_leaf)
    return u_leaf, u_leaf


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（参考实现），返回长度 N 的完整源向量估计。"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归接口：对信道 LLR 做 SC 译码。

    参数：
        llr_ch: 长度 N，LLR = ln P(y|0)/P(y|1)（与 channel.compute_llr 一致）
        frozen_bits: 长度 N，非 0 / True 表示冻结位

    返回：
        u_hat: 长度 N 的估计源向量（0/1）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits)
    frozen_ind = np.zeros(N, dtype=np.int8)
    frozen_ind[frozen_bits.astype(bool)] = 1

    u_hat, _ = _polar_decode_sc(llr_ch, frozen_ind)
    return np.round(u_hat).astype(int)


def precompute_sc_indices(N):
    """预计算辅助向量（与教材描述一致，供扩展/测试使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi & 1:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if (phi & 1) == 0:
            psi = phi
            layer = 0
            while (psi & 1) == 0 and psi != 0:
                bit_layers.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
