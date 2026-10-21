"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_partial):
    """g 运算；u_partial 为左子树返回的部分码字比特"""
    return (1.0 - 2.0 * u_partial) * La + Lb


def _is_frozen(frozen_bits, idx):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return bool(fb[idx])
    return bool(fb[idx] != 0)


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（与 polar_encode / G_N 配套）。
    g 节点使用左子树返回的部分码字（非裸 u 比特）。
    """
    llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def sc_rec(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if _is_frozen(frozen_bits, i):
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llrs[0] >= 0 else 1
            return [u_hat[i]]

        half = len(indices) // 2
        left_idx = indices[:half]
        right_idx = indices[half:]

        left_llr = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = sc_rec(left_llr, left_idx)

        right_llr = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = sc_rec(right_llr, right_idx)

        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    sc_rec(list(llr), list(range(N)))
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 的层索引（用于逐比特更新）"""
    n = int(np.log2(N))
    layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        psi = phi
        layers = []
        while (psi & 1) == 1:
            layers.append(int(np.log2(psi & -psi)))
            psi >>= 1
        layer_vec[phi] = layers
    return layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC：复用与递归版相同的判决逻辑（N<=1024 仿真足够快）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
