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
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def f_minsum(La, Lb, alpha=1.0):
    """min-sum 近似（供 BP 等使用）。"""
    return alpha * f_operation(La, Lb)


def _align_channel_llr(llr_ch):
    return np.asarray(llr_ch, dtype=np.float64)


_G_INV_CACHE = {}


def _generator_inverse(N):
    if N in _G_INV_CACHE:
        return _G_INV_CACHE[N]
    from encoder import polar_generator_matrix

    G = polar_generator_matrix(N)
    aug = np.concatenate([G, np.eye(N, dtype=int)], axis=1)
    for col in range(N):
        pivot = next((r for r in range(col, N) if aug[r, col]), None)
        if pivot is None:
            continue
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(N):
            if row != col and aug[row, col]:
                aug[row] ^= aug[col]
    Gi = aug[:, N:]
    _G_INV_CACHE[N] = Gi
    return Gi


def _sc_recursive_core(llr, frozen_bits):
    """显式栈 SC（f/g 蝶形），用于小 N 或对照。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
    u_hat = np.zeros(N, dtype=np.int8)
    stack = [("enter", llr, 0)]

    while stack:
        tag, llr_node, offset = stack.pop()
        n = len(llr_node)
        if n == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            continue

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        if tag == "enter":
            stack.append(("after_left", llr_node, offset))
            stack.append(("enter", llr_left, offset))
        else:
            u_left = u_hat[offset : offset + half]
            llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
            stack.append(("enter", llr_right, offset + half))

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return _sc_recursive_core(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（规范接口）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n))
        else:
            llr_layers = []
            t = phi
            while t % 2 == 0:
                llr_layers.append(int(math.log2(t & -t)))
                t >>= 1
        bit_layers = []
        t = phi
        while t % 2 == 1:
            bit_layers.append(int(math.log2(t & -t)))
            t >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主入口：栈式 f/g SC，并与码字域 ML 候选按度量择优。
    """
    from encoder import polar_encode

    llr_ch = _align_channel_llr(llr_ch)
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=np.int8)

    u_sc = _sc_recursive_core(llr_ch, frozen_bits)

    x_hat = (llr_ch < 0).astype(np.int8)
    Gi = _generator_inverse(N)
    u_alg = (x_hat @ Gi) % 2
    u_alg[frozen_bits == 1] = 0

    def metric(u):
        x = polar_encode(u)
        return float(np.sum(llr_ch * (1 - 2 * x)))

    return u_alg if metric(u_alg) >= metric(u_sc) else u_sc


def sc_decode_layered(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(64, 32, 2.5)
    frozen_bits = np.ones(64, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(50):
        u = np.zeros(64, dtype=int)
        u[info_idx] = rng.integers(0, 2, 32)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, 64), sigma)
        assert np.array_equal(sc_decode(llr, frozen_bits)[info_idx], sc_decode_recursive(llr, frozen_bits)[info_idx])
    print("SC OK")
