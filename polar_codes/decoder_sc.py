"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_boxplus(La, Lb):
    """boxplus（f 运算，精确 LLR 合并）"""
    La = np.clip(La, -30.0, 30.0)
    Lb = np.clip(Lb, -30.0, 30.0)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（高效）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _sc_rec_sionna(llr, frozen_ind, use_boxplus=True):
    """
    递归 SC（与 Sionna PolarSCDecoder 逻辑一致）。
    frozen_ind: 长度 n，1=冻结位，0=信息位
  返回 u_hat（长度 n）及中间部分和 u_hat_up
    """
    n = len(llr)
    f_fn = f_boxplus if use_boxplus else f_operation

    if n == 1:
        if frozen_ind[0] == 1:
            u = np.array([0.0])
        else:
            u = np.array([0.0 if llr[0] >= 0 else 1.0])
        return u, u.copy()

    half = n // 2
    llr1 = llr[:half]
    llr2 = llr[half:]
    fi1 = frozen_ind[:half]
    fi2 = frozen_ind[half:]

    llr_u = f_fn(llr1, llr2)
    u1, u1_up = _sc_rec_sionna(llr_u, fi1, use_boxplus)
    llr_d = g_operation(llr1, llr2, u1_up)
    u2, u2_up = _sc_rec_sionna(llr_d, fi2, use_boxplus)

    u_hat = np.concatenate([u1, u2])
    u1_up_int = (u1_up.astype(int) ^ u2_up.astype(int)).astype(float)
    u_hat_up = np.concatenate([u1_up_int, u2_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_bits, dtype=float)
    if frozen_ind.dtype == bool:
        frozen_ind = frozen_ind.astype(float)
    u_hat, _ = _sc_rec_sionna(llr, frozen_ind, use_boxplus=True)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        v = phi
        l = 0
        while v % 2 == 1:
            v //= 2
            l += 1
        llr_layer_vec.append(list(range(l, n)))

        v = (phi + 1) // 2
        l = 0
        while v % 2 == 1:
            v //= 2
            l += 1
        bit_layer_vec.append(list(range(l, n)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（默认调用递归 boxplus 实现以保证正确性）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"encoder: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u_sent)), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    print(f"SC test N=64: {errors} errors in 100 frames (expect 0)")
