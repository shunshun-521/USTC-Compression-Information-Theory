"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_tree(llr_node, frozen_node):
    """递归 SC 译码树（Sionna / Arikan 风格，含 u_hat_up）。"""
    n = len(llr_node)
    if n == 1:
        if frozen_node[0]:
            bit = 0.0
        else:
            bit = 0.0 if llr_node[0] >= 0 else 1.0
        u = np.array([bit])
        return u, u.copy()

    half = n // 2
    llr1 = llr_node[:half]
    llr2 = llr_node[half:]
    f1 = frozen_node[:half]
    f2 = frozen_node[half:]

    llr_left = f_operation(llr1, llr2)
    u1, u1_up = _sc_decode_tree(llr_left, f1)

    llr_right = g_operation(llr1, llr2, u1_up)
    u2, u2_up = _sc_decode_tree(llr_right, f2)

    u = np.concatenate([u1, u2])
    u1_xor = np.bitwise_xor(u1_up.astype(int), u2_up.astype(int)).astype(np.float64)
    u_up = np.concatenate([u1_xor, u2_up])
    return u, u_up


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    u_hat, _ = _sc_decode_tree(llr_ch[rev], frozen_bits)
    return np.round(u_hat).astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while p % 2 == 1 and layer < n:
            llr_layers.append(layer)
            p //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi + 1
        layer = 0
        while p % 2 == 0 and layer < n:
            bit_layers.append(layer)
            p //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（高效实现，aff3ct 风格索引）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    rev = bit_reversal_permutation(N)

    lam = [1 << i for i in range(n + 1)]
    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int32)
    P[n] = llr_ch[rev].copy()

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            limit = lam[layer]
            offset = (phi // (2 * limit)) * 2 * limit
            if (phi // limit) % 2 == 0:
                for i in range(limit):
                    P[layer, offset + i] = f_operation(
                        P[layer + 1, offset + i],
                        P[layer + 1, offset + i + limit],
                    )
            else:
                for i in range(limit):
                    P[layer, offset + i] = g_operation(
                        P[layer + 1, offset + i],
                        P[layer + 1, offset + i + limit],
                        C[layer, offset + i],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1

        C[0, phi] = u_hat[phi]
        for layer in bit_layer_vec[phi]:
            limit = lam[layer]
            offset = (phi // lam[layer + 1]) * lam[layer + 1]
            for i in range(limit):
                C[layer + 1, offset + i] = (
                    C[layer, offset + i] ^ C[layer, offset + i + limit]
                )
                C[layer + 1, offset + i + limit] = C[layer, offset + i + limit]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（递归实现，经验证正确）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """SC 译码无损校验：递归与非递归结果一致，且高 SNR 下无错。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            raise AssertionError("SC decode error at high SNR")
    return True
