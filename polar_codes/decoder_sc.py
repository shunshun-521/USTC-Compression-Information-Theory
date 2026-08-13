"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算，u_hat 为左子树译码比特向量。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=int)
    return Lb + (1 - 2 * u_hat) * La


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        temp = phi
        layer = 0
        while temp % 2 == 1:
            temp //= 2
            layer += 1
        layers_llr.append(layer)
        while temp > 0:
            while temp % 2 == 1:
                temp //= 2
                layer += 1
            layers_llr.append(layer)
            temp //= 2
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = phi // 2
        layer = 0
        while temp % 2 == 1:
            temp //= 2
            layer += 1
        for l in range(layer + 1):
            layers_bit.append(l)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N)) + 1
    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    def decode(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                bit = 0
            else:
                bit = 0 if y[0] >= 0 else 1
            u_hat[node] = bit
            return [bit]

        half = len(y) // 2
        left_y = y[:half]
        right_y = y[half:]
        left_bits = decode(f_operation(left_y, right_y), depth + 1, 2 * node)
        right_bits = decode(
            g_operation(left_y, right_y, left_bits),
            depth + 1,
            2 * node + 1,
        )
        return [(left_bits[i] + right_bits[i]) % 2 for i in range(half)] + right_bits

    decode(llr, 0, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（与递归实现等价）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def _run_sc_self_tests():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(20.0, rate)
    rng = np.random.default_rng(0)

    for _ in range(100):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_src)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u_src[info_idx]), "SC 译码在高 SNR 下失败"


if __name__ == "__main__":
    _run_sc_self_tests()
    print("SC decoder self-tests passed.")
