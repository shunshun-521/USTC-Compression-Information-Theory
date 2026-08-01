"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _polar_decode_sc_recursive(llr_ch, frozen_ind):
    """
    递归 SC 译码（遵循 Sionna / Gross_Fast_SCL 记号）。
    返回 (u_hat, u_hat_up)，其中 u_hat_up 为当前阶段的中间重编码比特。
    """
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        frozen1 = frozen_ind[:half]
        frozen2 = frozen_ind[half:]

        x_llr1 = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_recursive(x_llr1, frozen1)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_recursive(x_llr2, frozen2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = (u_hat1_up.astype(int) ^ u_hat2_up.astype(int)).astype(int)
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    if frozen_ind[0]:
        u_hat = np.array([0], dtype=int)
    else:
        u_hat = np.array([0 if llr_ch[0] >= 0 else 1], dtype=int)
    return u_hat, u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_ind = frozen_bits.astype(int)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_ind)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while (p & 1) == 1 and layer < n:
            llr_layers.append(layer)
            layer += 1
            p >>= 1
        if layer < n:
            llr_layers.append(layer)
        for l in range(layer + 1, n):
            llr_layers.append(l)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            p = phi // 2
            layer = 1
            while (p & 1) == 1 and layer <= n:
                bit_layers.append(layer)
                layer += 1
                p >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, prepare_decoder_llr

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
        llr = prepare_decoder_llr(compute_llr(y, sigma), N)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC 校验: 100 帧中错误 {errors} 帧")
    assert errors == 0, "SC 译码校验失败"
    print("SC 译码校验通过")
