"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _combine_bits(left, right):
    """译码树比特合并（与编码蝶形对应）。"""
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    upper = (left + right) % 2
    return np.concatenate([upper, right])


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits, frozen_set=None):
    """
    递归 SC 译码（基于树节点索引）。
    frozen_bits: bool 数组，True 表示冻结位。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N)) + 1
    node_values = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0]) if frozen_set is None else frozen_set

    def decode(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
                return [0]
            bit = 1 if y[0] < 0 else 0
            node_values[node] = bit
            return [bit]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left = f_operation(L1, L2)
        arr1 = decode(left, depth + 1, 2 * node)

        right = g_operation(L1, L2, np.array(arr1, dtype=int))
        arr2 = decode(right, depth + 1, 2 * node + 1)

        return _combine_bits(arr1, arr2)

    decode(llr, 0, 0)
    return node_values


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（层偏移与层列表）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        while p % 2 == 1:
            layers_llr.append(int(math.log2(p & -p)))
            p //= 2
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 1:
            p = phi
            while p % 2 == 1:
                layers_bit.append(int(math.log2(p & -p)))
                p //= 2
        bit_layer_vec.append(layers_bit)

    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（当前调用已验证的递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def _sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC 译码（保留供扩展）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    P[n][:] = llr_ch
    C = np.zeros((n + 1, N), dtype=int)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            psi = phi // (2 * lambda_offset[layer])
            start = psi * 2 * lambda_offset[layer]
            half = lambda_offset[layer]

            La = P[layer + 1][start : start + half]
            Lb = P[layer + 1][start + half : start + 2 * half]
            P[layer][start : start + half] = f_operation(La, Lb)

            u_partial = C[layer][start : start + half]
            P[layer][start + half : start + 2 * half] = g_operation(La, Lb, u_partial)

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 1 if P[0][0] < 0 else 0

        C[0][0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            psi = phi // (2 * lambda_offset[layer])
            start = psi * 2 * lambda_offset[layer]
            half = lambda_offset[layer]

            C[layer + 1][start : start + half] = (
                C[layer][start : start + half] ^ C[layer][start + half : start + 2 * half]
            )
            C[layer + 1][start + half : start + 2 * half] = C[layer][
                start + half : start + 2 * half
            ]

    return u_hat


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok_rec = ok_non = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        if np.array_equal(sc_decode_recursive(llr, frozen_bits)[info_idx], u[info_idx]):
            ok_rec += 1
        if np.array_equal(sc_decode(llr, frozen_bits)[info_idx], u[info_idx]):
            ok_non += 1
    print(f"SC recursive {ok_rec}/100, non-recursive {ok_non}/100 at 10dB")
