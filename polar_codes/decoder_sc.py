"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


def prepare_llr_for_decode(llr_ch, N):
    """将信道 LLR 置换为极化信道顺序（与比特倒序编码配套）。"""
    from encoder import bit_reversal_permutation

    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n_depth = int(np.log2(N)) + 1
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = [0] * N

    def decode(y, depth, node):
        if depth == n_depth - 1:
            if node in frozen_set:
                node_values[node] = 0
            else:
                node_values[node] = 1 if y[0] < 0 else 0
            return [node_values[node]]

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left = f_operation(np.array(l1), np.array(l2)).tolist()
        arr1 = decode(left, depth + 1, 2 * node)
        right = g_operation(np.array(l1), np.array(l2), np.array(arr1)).tolist()
        arr2 = decode(right, depth + 1, 2 * node + 1)
        return [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))] + arr2

    decode(llr.tolist(), 0, 0)
    return np.array(node_values, dtype=int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        temp = phi
        for layer in range(n):
            if temp & 1:
                bit_layers.append(layer)
            else:
                llr_layers.append(layer)
            temp >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（显式栈实现，与递归版本等价）。
    """
    llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr)
    n_depth = int(np.log2(N)) + 1
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    node_values = [0] * N
    returns = {}

    stack = [("start", 0, 0, llr.tolist())]

    while stack:
        kind, depth, node, y = stack.pop()

        if kind == "start":
            if depth == n_depth - 1:
                if node in frozen_set:
                    node_values[node] = 0
                else:
                    node_values[node] = 1 if y[0] < 0 else 0
                returns[(depth, node)] = [node_values[node]]
                continue

            half = len(y) // 2
            left = f_operation(np.array(y[:half]), np.array(y[half:])).tolist()
            stack.append(("after_left", depth, node, y))
            stack.append(("start", depth + 1, 2 * node, left))

        elif kind == "after_left":
            half = len(y) // 2
            l1, l2 = y[:half], y[half:]
            arr1 = returns.pop((depth + 1, 2 * node))
            right = g_operation(np.array(l1), np.array(l2), np.array(arr1)).tolist()
            stack.append(("after_right", depth, node, arr1))
            stack.append(("start", depth + 1, 2 * node + 1, right))

        elif kind == "after_right":
            arr1 = y
            arr2 = returns.pop((depth + 1, 2 * node + 1))
            returns[(depth, node)] = [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))] + arr2

    return np.array(node_values, dtype=int)


def sc_llr_at_phi(llr_ch, decided_bits, phi, frozen_bits):
    """
    给定前缀 decided_bits[0:phi]，计算第 phi 个比特的 LLR。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n_depth = int(np.log2(N)) + 1

    def decode_left(y, depth, node):
        if depth == n_depth - 1:
            return [int(decided_bits[node])]
        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left = f_operation(np.array(l1), np.array(l2)).tolist()
        arr1 = decode_left(left, depth + 1, 2 * node)
        right = g_operation(np.array(l1), np.array(l2), np.array(arr1)).tolist()
        arr2 = decode_left(right, depth + 1, 2 * node + 1)
        return [(arr1[i] + arr2[i]) % 2 for i in range(len(arr1))] + arr2

    def get_llr(y, depth, node):
        if depth == n_depth - 1:
            return y[0]
        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        left = f_operation(np.array(l1), np.array(l2)).tolist()
        left_end = node * len(y) + half
        if phi < left_end:
            return get_llr(left, depth + 1, 2 * node)
        arr1 = decode_left(left, depth + 1, 2 * node)
        right = g_operation(np.array(l1), np.array(l2), np.array(arr1)).tolist()
        return get_llr(right, depth + 1, 2 * node + 1)

    return get_llr(llr_ch.tolist(), 0, 0)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = prepare_llr_for_decode(compute_llr(y, sigma), N)
        u_rec = sc_decode(llr, frozen_bits)
        u_rec_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_rec_r), "SC implementations differ"
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            errors += 1
    print(f"SC test: {errors}/100 errors at Eb/N0=10dB")
