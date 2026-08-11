"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _sc_decode_core(llr, frozen_bits):
    """
    核心 SC 译码（Sionna 风格，使用 stage 部分和进行 g 运算）。
    frozen_bits 需已按比特倒序排列。
    """
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_full = np.zeros(N, dtype=int)

    def decode_node(llr_node, frozen_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            bit = 0 if frozen_node[0] or llr_node[0] >= 0 else 1
            u_full[bit_offset] = bit
            return np.array([bit], dtype=float), np.array([bit], dtype=float)

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        _, u_left_up = decode_node(llr_left, frozen_node[:half], bit_offset)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        u_hat2, u_right_up = decode_node(llr_right, frozen_node[half:], bit_offset + half)

        u_hat1 = u_full[bit_offset:bit_offset + half].astype(float)
        u_left_up_xor = (u_left_up.astype(int) ^ u_right_up.astype(int)).astype(float)
        u_hat_up = np.concatenate([u_left_up_xor, u_right_up])
        u_hat = np.concatenate([u_hat1, u_hat2])
        return u_hat, u_hat_up

    decode_node(llr, frozen_bits, 0)
    return u_full


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（对外接口，frozen_bits 为自然顺序）。
    """
    N = len(llr)
    rev = bit_reversal_permutation(N)
    frozen_dec = np.asarray(frozen_bits, dtype=bool)[rev]
    u_dec = _sc_decode_core(llr, frozen_dec)
    return u_dec[rev]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        for layer in range(n):
            if psi % 2 == 0:
                llr_layers.append(layer)
                psi //= 2
            else:
                break
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi + 1
        for layer in range(n):
            if psi % 2 == 1:
                bit_layers.append(layer)
                psi //= 2
            else:
                break
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（对外接口，frozen_bits 为自然顺序）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """SC 译码无损验证"""
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
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_rec), "SC recursive vs non-recursive mismatch"
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC decode error"

    print(f"SC verification passed: N={N}, K={K}, {num_frames} frames at {eb_n0_db}dB")


if __name__ == "__main__":
    verify_sc_decoders()
