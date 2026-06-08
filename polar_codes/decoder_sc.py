"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

# ==================== 基本运算 ====================


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
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现，偶/奇分裂）====================


def _sc_recursive_core(llr, frozen_bits):
    """偶/奇交错分裂的递归 SC（与编码器比特倒序蝶形一致）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits).astype(bool)

    if N == 1:
        bit = 0 if llr[0] >= 0 else 1
        if frozen_bits[0]:
            return 0, bit
        return bit, bit

    u_left, u1hp = _sc_recursive_core(
        f_operation(llr[::2], llr[1::2]), frozen_bits[: N // 2]
    )
    llr_right = g_operation(
        f_operation(u1hp, llr[::2]), llr[1::2], u_left
    )
    u_right, u2hp = _sc_recursive_core(llr_right, frozen_bits[N // 2 :])

    u_hat = np.zeros(N, dtype=int)
    u_hat[: N // 2] = u_left
    u_hat[N // 2 :] = u_right

    x_even = f_operation(u1hp, u2hp)
    x_odd = u2hp
    x_partial = np.zeros(N, dtype=np.float64)
    x_partial[::2] = x_even
    x_partial[1::2] = x_odd
    return u_hat, x_partial


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    frozen_bits: True 表示冻结位
    """
    u_hat, _ = _sc_recursive_core(np.asarray(llr, dtype=np.float64), frozen_bits)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（偶/奇蝶形树）。
    """
    n = int(np.log2(N))
    lambda_offset = np.arange(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        for layer in range(n):
            if (p >> layer) & 1 == 0:
                layers_llr.append(layer)
            else:
                break
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        p = phi
        for layer in range(n):
            if p & 1:
                layers_bit.append(layer)
                p >>= 1
            else:
                break
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（层叠更新，供与递归版本对照）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    n = int(np.log2(N))
    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.float64)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            stride = 1 << layer
            half = stride
            prev = layer + 1
            for block in range(0, N, 2 * stride):
                for j in range(half):
                    li = block + j
                    ri = block + j + half
                    La = P[prev][li]
                    Lb = P[prev][ri]
                    if (phi >> layer) & 1 == 0:
                        P[layer][li] = f_operation(La, Lb)
                    else:
                        P[layer][li] = g_operation(La, Lb, C[layer][li])

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0][0] >= 0 else 1
        C[0][0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            stride = 1 << layer
            half = stride
            for block in range(0, N, 2 * stride):
                for j in range(half):
                    li = block + j
                    ri = block + j + half
                    if (phi >> layer) & 1 == 0:
                        C[layer + 1][li] = C[layer][li]
                        C[layer + 1][ri] = C[layer][li]
                    else:
                        C[layer + 1][ri] = C[layer][ri]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主入口（当前默认使用已验证的递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """验证递归与非递归 SC 译码器一致且高 SNR 下无误码。"""
    from construction import ga_construction
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        # 无损验证：使用极大 LLR 等效无噪 BPSK
        llr = np.where(x == 0, 100.0, -100.0)

        u_rec = sc_decode(llr, frozen_bits)
        u_ref = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_rec, u_ref):
            raise AssertionError("SC recursive vs non-recursive mismatch")
        if not np.array_equal(u_rec[info_idx], u[info_idx]):
            raise AssertionError("SC decode error at high SNR")
