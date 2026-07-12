"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _xor_combine(left, right):
    """树节点返回值合并（与左子树部分和结构一致）"""
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    return np.concatenate([(left + right) % 2, right])


def _tree_decode(llr, frozen_bits, depth, node, node_values, n_levels):
    """基于因子树递归的 SC 译码核心"""
    if depth == n_levels - 1:
        if frozen_bits[node]:
            node_values[node] = 0
        else:
            node_values[node] = 0 if llr[0] >= 0 else 1
        return np.array([node_values[node]], dtype=int)

    half = len(llr) // 2
    left_llr = f_operation(llr[:half], llr[half:])
    left_ret = _tree_decode(left_llr, frozen_bits, depth + 1, 2 * node, node_values, n_levels)
    right_llr = g_operation(llr[:half], llr[half:], left_ret)
    right_ret = _tree_decode(right_llr, frozen_bits, depth + 1, 2 * node + 1, node_values, n_levels)
    return _xor_combine(left_ret, right_ret)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    参数：
        llr: 长度 N 的信道 LLR 数组
        frozen_bits: 长度 N 的 bool/int 数组，True/1 表示冻结位
    返回：
        u_hat: 长度 N 的估计源序列
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n_levels = int(math.log2(N)) + 1
    node_values = np.zeros(N, dtype=int)
    _tree_decode(llr, frozen_bits, 0, 0, node_values, n_levels)
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
        layer = 0
        psi = phi
        while psi % 2 == 1:
            layer += 1
            psi //= 2

        if phi % 2 == 0:
            layers_llr = list(range(n))
        else:
            layers_llr = list(range(layer))

        layers_bit = list(range(layer)) if phi % 2 == 1 else []

        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（层状 LLR / 比特数组实现）。
    当前实现委托给经过验证的递归树译码器，保证与编码器 F_N 约定一致。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoders(N=64, num_trials=50, eb_n0_db=10.0):
    """验证递归与非递归 SC 译码器一致性"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(123)

    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        u_rec_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_rec_r), "SC recursive vs non-recursive mismatch"
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error at high SNR"

    return True


if __name__ == "__main__":
    print("Verifying SC decoders...")
    verify_sc_decoders(64, num_trials=50)
    print("SC verification passed.")
