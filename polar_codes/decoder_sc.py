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
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb（向量化）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_tree(llr_ch, frozen_bits):
    """
    SC 译码核心：g 运算使用左子树的部分和 beta（而非单个 u 比特）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen = np.asarray(frozen_bits, dtype=int).astype(bool)
    n = len(llr_ch)
    u_hat = np.zeros(n, dtype=int)

    def decode_node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen[idx]:
                u_hat[idx] = 0
                return np.array([0], dtype=int)
            u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = length // 2
        upper = f_operation(llr_node[:half], llr_node[half:])
        beta_upper = decode_node(upper, base, half)
        lower_llr = g_operation(llr_node[:half], llr_node[half:], beta_upper)
        beta_lower = decode_node(lower_llr, base + half, half)
        return np.concatenate([beta_upper ^ beta_lower, beta_lower])

    decode_node(llr_ch, 0, n)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    return _sc_decode_tree(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        tmp = phi
        for i in range(n):
            if tmp % 2 == 0:
                llr_layers.append(i)
                tmp >>= 1
            else:
                break
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        tmp = phi
        for i in range(n):
            if tmp % 2 == 1:
                bit_layers.append(i)
                tmp >>= 1
            else:
                break
        bit_layer_vec.append(bit_layers)

    lambda_offset = np.array([2 ** i for i in range(n + 1)], dtype=int)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（显式栈模拟树遍历，避免深层递归）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen = np.asarray(frozen_bits, dtype=int).astype(bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    # 栈元素：(llr_node, base, length, state, beta_upper)
    # state: 0=enter, 1=after_left, 2=after_right
    stack = [(llr_ch, 0, N, 0, None)]
    beta_results = []

    while stack:
        llr_node, base, length, state, saved_beta = stack.pop()

        if length == 1:
            idx = base
            if frozen[idx]:
                u_hat[idx] = 0
                beta = np.array([0], dtype=int)
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
                beta = np.array([u_hat[idx]], dtype=int)
            beta_results.append(beta)
            continue

        if state == 0:
            half = length // 2
            upper = f_operation(llr_node[:half], llr_node[half:])
            stack.append((llr_node, base, length, 1, None))
            stack.append((upper, base, half, 0, None))
        elif state == 1:
            beta_upper = beta_results.pop()
            half = length // 2
            lower_llr = g_operation(llr_node[:half], llr_node[half:], beta_upper)
            stack.append((llr_node, base, length, 2, beta_upper))
            stack.append((lower_llr, base + half, half, 0, None))
        else:
            beta_upper = saved_beta
            beta_lower = beta_results.pop()
            beta_results.append(np.concatenate([beta_upper ^ beta_lower, beta_lower]))

    return u_hat


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    for _ in range(50):
        info_bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.001)
        u1 = sc_decode_recursive(llr, frozen)
        u2 = sc_decode(llr, frozen)
        assert np.array_equal(u1, u2)
        assert np.array_equal(u1, u)

    print("SC decoder tests passed.")
