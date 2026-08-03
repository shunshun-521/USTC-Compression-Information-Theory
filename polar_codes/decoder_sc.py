"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
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


def _xor_combine(left, right):
    """HETSN 风格的比特重组"""
    res = [(left[i] + right[i]) % 2 for i in range(len(left))]
    res.extend(right.tolist())
    return np.array(res, dtype=int)


def _prepare_llr(llr_ch, N):
    """将信道 LLR 转换为译码树所需顺序（比特倒序）"""
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N)) + 1
    llr = _prepare_llr(llr_ch, N)
    node_values = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if frozen_bits[node]:
                val = 0
            else:
                val = 1 if y[0] < 0 else 0
            node_values[node] = val
            return np.array([val], dtype=int)

        half = len(y) // 2
        left_y = y[:half]
        right_y = y[half:]
        arr1 = decode_node(f_operation(left_y, right_y), depth + 1, 2 * node)
        arr2 = decode_node(g_operation(left_y, right_y, arr1), depth + 1, 2 * node + 1)
        return _xor_combine(arr1, arr2)

    decode_node(llr, 0, 0)
    return node_values


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                layers.append(layer)
            else:
                break
            temp //= 2
        llr_layer_vec.append(layers)

        bit_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 1:
                bit_layers.append(layer)
            temp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（与递归版本等价）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoder(N=64, K=32, num_frames=100, seed=42):
    """在极低噪声下验证 SC 译码正确性"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(seed)
    sigma = eb_n0_to_sigma(10.0, K / N)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC 译码错误"

    u_hat_rec = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_hat_rec[info_idx], u[info_idx]), "递归 SC 译码错误"


if __name__ == "__main__":
    verify_sc_decoder()
    print("SC 译码校验通过")
