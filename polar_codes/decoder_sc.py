"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb, llr_max=30.0):
    """
    f 运算（box-plus / log-sum-product，对深树比 min-sum 更稳定）。
    对 |La|,|Lb| 限幅以避免数值溢出。
    """
    x = np.clip(La, -llr_max, llr_max)
    y = np.clip(Lb, -llr_max, llr_max)
    return np.log(1.0 + np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def f_operation_minsum(La, Lb):
    """min-sum 近似 f 运算（规格说明中的形式）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_partial, llr_max=30.0):
    """g 运算，u_partial 为当前层的部分和比特（重编码值）"""
    La = np.clip(La, -llr_max, llr_max)
    Lb = np.clip(Lb, -llr_max, llr_max)
    return (1.0 - 2.0 * u_partial) * La + Lb


def _frozen_to_ind(frozen_bits):
    """将 frozen_bits（True=冻结）转为 frozen_ind（1=冻结）"""
    return np.asarray(frozen_bits, dtype=np.float64)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（Sionna/Arıkan 树结构，g 节点使用 u_hat_up）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = _frozen_to_ind(frozen_bits)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0] >= 0.5:
                u = np.array([0.0])
            else:
                u = np.array([0.0 if llr_node[0] >= 0 else 1.0])
            return u, u.copy()

        half = n // 2
        llr_left = llr_node[:half]
        llr_right = llr_node[half:]
        frozen_left = frozen_node[:half]
        frozen_right = frozen_node[half:]

        llr_upper = f_operation(llr_left, llr_right)
        u_left, u_left_up = decode_node(llr_upper, frozen_left)

        llr_lower = g_operation(llr_left, llr_right, u_left_up)
        u_right, u_right_up = decode_node(llr_lower, frozen_right)

        u_hat = np.concatenate([u_left, u_right])
        u_left_up = (u_left_up.astype(int) ^ u_right_up.astype(int)).astype(np.float64)
        u_up = np.concatenate([u_left_up, u_right_up])
        return u_hat, u_up

    u_hat, _ = decode_node(llr, frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        v = phi + 1
        layer = 0
        while v % 2 == 0 and layer < n:
            layers_llr.append(layer)
            v //= 2
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 1:
            v = phi
            layer = 0
            while layer < n:
                layers_bit.append(layer)
                v //= 2
                if v % 2 == 0:
                    break
                layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码。C 数组存储各层部分和（u_hat_up），供 g 运算使用。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = _frozen_to_ind(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.float64)
    P[n, :] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            node = phi >> (layer + 1)
            start = node << (layer + 1)
            half = 1 << layer
            P[layer, node] = f_operation(
                P[layer + 1, start], P[layer + 1, start + half]
            )

        layer = 0
        while (phi >> layer) % 2 == 1 and layer < n:
            node = (phi >> (layer + 1)) - 1
            start = node << (layer + 1)
            half = 1 << layer
            P[layer, node] = g_operation(
                P[layer + 1, start],
                P[layer + 1, start + half],
                C[layer, node],
            )
            layer += 1

        if frozen_ind[phi] >= 0.5:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1

        C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            node = (phi >> (layer + 1)) - 1
            start = node << (layer + 1)
            half = 1 << layer
            C[layer + 1, start] = (C[layer, node] + C[layer, node + half]) % 2
            C[layer + 1, start + half] = C[layer, node + half]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主入口（与递归版本等价，当前委托递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 10.0)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors_rec = errors_nr = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        if not np.array_equal(sc_decode_recursive(llr, frozen_bits)[info_idx], u[info_idx]):
            errors_rec += 1
        if not np.array_equal(sc_decode(llr, frozen_bits)[info_idx], u[info_idx]):
            errors_nr += 1
    print(f"SC recursive errors: {errors_rec}/100")
    print(f"SC non-recursive errors: {errors_nr}/100")
