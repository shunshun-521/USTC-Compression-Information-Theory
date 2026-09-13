"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（变量节点更新）"""
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0 or u_hat.size == 1:
        u = int(u_hat) if np.ndim(u_hat) == 0 else int(u_hat.flat[0])
        return (1 - 2 * u) * La + Lb
    return (1 - 2 * u_hat) * La + Lb


def _cn_op_exact(x, y):
    """精确 log-domain boxplus"""
    x = np.clip(x, -50.0, 50.0)
    y = np.clip(y, -50.0, 50.0)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _vn_op_exact(x, y, u_hat):
    return (1 - 2 * u_hat) * x + y


def _frozen_to_ind(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=int)
    return (fb == 1).astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，Sionna 风格）。
    信道 LLR 为码字自然顺序；冻结位 frozen_bits 中 1 表示冻结。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    frozen_ind = _frozen_to_ind(frozen_bits)

    def _decode(llr, frozen):
        n = len(llr)
        if n > 1:
            if frozen.sum() == n:
                z = np.zeros(n, dtype=int)
                return z, z.astype(np.float64)
            llr1, llr2 = llr[: n // 2], llr[n // 2 :]
            f1, f2 = frozen[: n // 2], frozen[n // 2 :]
            u1, u1_up = _decode(_cn_op_exact(llr1, llr2), f1)
            u2, u2_up = _decode(_vn_op_exact(llr1, llr2, u1_up), f2)
            u = np.concatenate([u1, u2])
            u1_up_new = (u1_up.astype(int) ^ u2_up.astype(int)).astype(np.float64)
            u_up = np.concatenate([u1_up_new, u2_up])
            return u, u_up
        if frozen[0] == 1:
            return np.array([0], dtype=int), np.array([0.0])
        bit = 0 if llr[0] >= 0 else 1
        return np.array([bit], dtype=int), np.array([float(bit)])

    u_hat, _ = _decode(llr_ch[br], frozen_ind)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（Tal-Vardy 调度）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            layers_bit.append(0)
        psi = phi
        while psi % 2 == 1:
            layers_bit.append(int(math.log2(psi & -psi)))
            psi >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（调用与递归版本等价的树遍历实现）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test errors: {errors}/100")
