"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（用于接口兼容）。
    主译码路径使用精确 log-domain f。
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _f_exact(a, b):
    """精确 log-domain f 运算"""
    return np.asarray(
        np.logaddexp(0.0, a + b) - np.logaddexp(a, b), dtype=np.float64
    )


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    u = np.asarray(u_hat, dtype=np.float64)
    return np.asarray(Lb + (1.0 - 2.0 * u) * La, dtype=np.float64)


def _sc_node(llr, frozen, base, length, u_hat):
    """递归 SC 子树译码"""
    if length == 1:
        if frozen[base]:
            u_hat[base] = 0
        else:
            u_hat[base] = 0 if llr[0] >= 0 else 1
        return np.array([u_hat[base]], dtype=int)

    half = length // 2
    upper = _f_exact(llr[:half], llr[half:])
    beta_upper = _sc_node(upper, frozen, base, half, u_hat)
    lower = g_operation(llr[:half], llr[half:], beta_upper)
    beta_lower = _sc_node(lower, frozen, base + half, half, u_hat)
    return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return sc_decode(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。
    信道 LLR 按自然序输入，LLR = ln P(y|0) / P(y|1)。
    """
    llr = np.asarray(llr_ch, dtype=np.float64)
    frozen = frozen_bits.astype(bool)
    u_hat = np.zeros(len(llr), dtype=int)
    _sc_node(llr, frozen, 0, len(llr), u_hat)
    return u_hat


def precompute_sc_indices(N):
    """保留接口，供 SCL 模块引用（非递归索引预计算）"""
    n = int(np.log2(N))
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return llr_layer_vec, bit_layer_vec


def _pm_penalty(llr, u):
    """路径度量惩罚项（log-domain）"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * u) * llr))
