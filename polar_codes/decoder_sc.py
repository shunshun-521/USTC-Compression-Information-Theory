"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 box-plus f 运算。"""
    La = np.clip(La, -30, 30)
    Lb = np.clip(Lb, -30, 30)
    return np.sign(La) * np.sign(Lb) * np.log(
        (1 + np.exp(-np.abs(La + Lb))) / (1 + np.exp(-np.abs(La - Lb)))
    )


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_sionna(llr_ch, frozen_bits, use_exact=False):
    """
    SC 译码（遵循 Sionna / Arikan 递归树，VN 使用 u_hat_up）。
    llr_ch: 信道 LLR（正表示倾向 bit 0）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    cn = f_operation_exact if use_exact else f_operation

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                bit = 0.0
            else:
                bit = 0.5 * (1.0 - np.sign(llr_node[0]))
                if bit == 0.5:
                    bit = 1.0
            u = np.array([bit])
            return u, u.copy()

        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        f1 = frozen_node[:half]
        f2 = frozen_node[half:]

        llr_left = cn(llr1, llr2)
        u1, u1_up = decode_node(llr_left, f1)

        llr_right = g_operation(llr1, llr2, u1_up)
        u2, u2_up = decode_node(llr_right, f2)

        u = np.concatenate([u1, u2])
        u1_up_int = np.bitwise_xor(u1_up.astype(int), u2_up.astype(int)).astype(np.float64)
        u_up = np.concatenate([u1_up_int, u2_up])
        return u, u_up

    u_hat, _ = decode_node(llr_ch, frozen_bits)
    return np.round(u_hat).astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return _sc_decode_sionna(llr_ch[rev], frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
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


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（对信道 LLR 做比特倒序后译码）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """SC 译码无损校验"""
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
            raise AssertionError("SC decode error")
    return True
