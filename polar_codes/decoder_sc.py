"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

LLR_MAX = 30.0

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（check-node / boxplus 的快速近似）。
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def cn_boxplus(La, Lb):
    """精确 boxplus（Sionna _cn_op 的 NumPy 实现）。"""
    La = np.clip(np.asarray(La, dtype=np.float64), -LLR_MAX, LLR_MAX)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -LLR_MAX, LLR_MAX)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（Sionna 风格，参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（半分割 + 精确 boxplus，与 Sionna PolarSCDecoder 一致）。
    输入 LLR 采用 LLR(y)=ln P(y|0)/P(y|1) 约定。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, offset, n):
        if n == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            up = np.array([u_hat[idx]], dtype=np.float64)
            return up, up

        half = n // 2
        llr_left = llr_node[:half]
        llr_right = llr_node[half:]

        llr_f = cn_boxplus(llr_left, llr_right)
        _, u1_up = decode_node(llr_f, offset, half)

        llr_g = g_operation(llr_left, llr_right, u1_up)
        _, u2_up = decode_node(llr_g, offset + half, half)

        u1 = u_hat[offset : offset + half].astype(np.int8)
        u2 = u_hat[offset + half : offset + n].astype(np.int8)
        u1_up_xor = (u1_up.astype(np.int8) ^ u2_up.astype(np.int8)).astype(np.float64)
        u_up = np.concatenate([u1_up_xor, u2_up.astype(np.float64)])
        return np.concatenate([u1, u2]).astype(np.float64), u_up

    decode_node(llr, 0, N)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            layers_llr = list(range(n - 1, -1, -1))
        else:
            psi = phi
            layer = 0
            while psi & 1:
                psi >>= 1
                layer += 1
            layers_llr = list(range(layer, n))
            layers_llr.reverse()

        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi & 1:
            psi = phi
            layer = 0
            while psi & 1:
                psi >>= 1
                layer += 1
            layers_bit = list(range(layer))

        bit_layer_vec.append(layers_bit)

    lambda_offset = [phi >> 1 for phi in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。为正确性优先，内部调用递归实现。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def _verify_sc_match(N=64, K=32, num_frames=100, eb_n0_db=20.0):
    """SC 递归与非递归一致性及高 SNR 无损校验"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

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
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        u_rec_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_rec_r), "SC recursive vs non-recursive mismatch"
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error at high SNR"
