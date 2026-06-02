"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

LLR_MAX = 30.0

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus 可选）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def cn_op_boxplus(x, y):
    """Check-node 更新（boxplus，与 Sionna 一致）"""
    x = np.clip(x, -LLR_MAX, LLR_MAX)
    y = np.clip(y, -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def g_operation(La, Lb, u_hat):
    """g 运算（VN 更新）"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================


def _polar_decode_sc_recursive(llr_ch, frozen_ind, use_boxplus=True):
    """
    递归 SC 译码（树形结构，g 节点使用上层再编码比特 u_hat_up）。
    frozen_ind: 长度 n，1=冻结，0=信息。
    """
    n = len(llr_ch)
    cn = cn_op_boxplus if use_boxplus else f_operation

    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        fz1 = frozen_ind[:half]
        fz2 = frozen_ind[half:]

        x_llr1 = cn(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_recursive(x_llr1, fz1, use_boxplus)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_recursive(x_llr2, fz2, use_boxplus)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = (u_hat1_up.astype(int) ^ u_hat2_up.astype(int)).astype(float)
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    # 叶节点
    if frozen_ind[0] == 1:
        u_hat = np.array([0.0])
    else:
        if llr_ch[0] >= 0:
            u_hat = np.array([0.0])
        else:
            u_hat = np.array([1.0])
    return u_hat, u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    frozen_ind = frozen_bits.astype(int)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_ind, use_boxplus=True)
    return u_hat.astype(int)


# ==================== 非递归 SC 译码 ====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        p = phi
        t = 0
        while p % 2 == 1:
            p >>= 1
            t += 1
        llr_layer_vec.append(list(range(t, n)))

        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            layers = []
            p, l = phi, 0
            while p % 2 == 1:
                layers.append(l)
                p >>= 1
                l += 1
            bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（默认调用递归 boxplus 实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


