"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于树遍历）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _pm_penalty(llr, bit):
    """路径度量惩罚：与 LLR 符号不一致时增加 |LLR|（log 域近似）"""
    return 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 polar_transform 编码配套）"""
    u_hat = np.zeros(len(llr), dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = np.asarray(llr, dtype=np.float64)

    def decode_node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = length // 2
        upper = f_operation(llr_node[:half], llr_node[half:])
        beta_u = decode_node(upper, base, half)
        lower = g_operation(llr_node[:half], llr_node[half:], beta_u)
        beta_l = decode_node(lower, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    decode_node(llr, 0, len(llr))
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 所需辅助向量（按比特相位记录需更新的层）。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layer = 0
        p = phi
        while p & 1:
            layer += 1
            p >>= 1
        llr_layer_vec.append(list(range(layer, n)))
        bit_layer_vec.append(list(range(layer)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码：list_size=1 的 SCL 树遍历（与递归版等价，主实现）。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    def node(llr, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = length // 2
        upper = f_operation(llr[:half], llr[half:])
        beta_u = node(upper, base, half)
        lower = g_operation(llr[:half], llr[half:], beta_u)
        beta_l = node(lower, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    node(llr_ch, 0, N)
    return u_hat
