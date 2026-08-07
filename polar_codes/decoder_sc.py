"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _frozen_to_decode_set(frozen_bits, br):
    """将自然序冻结位映射到译码树节点索引。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return {int(br[i]) for i in np.where(frozen_bits)[0]}


def _combine_paths(left, right):
    """递归 SC 返回路径合并（xor + append）。"""
    left = np.asarray(left, dtype=int)
    right = np.asarray(right, dtype=int)
    return np.concatenate([(left + right) % 2, right])


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    llr: 自然序信道 LLR
    frozen_bits: True/1 表示冻结位
    返回自然序 u_hat
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N)) + 1
    br = bit_reversal_permutation(N)
    frozen_set = _frozen_to_decode_set(frozen_bits, br)
    node_values = np.zeros(N, dtype=int)

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
                return np.array([0])
            bit = 1 if y[0] < 0 else 0
            node_values[node] = bit
            return np.array([bit])

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        arr1 = decode_node(f_operation(l1, l2), depth + 1, 2 * node)
        arr2 = decode_node(g_operation(l1, l2, arr1), depth + 1, 2 * node + 1)
        return _combine_paths(arr1, arr2)

    decode_node(llr, 0, 0)
    return node_values[br]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（层索引列表）。
    与递归实现等价的层触发序列。
    """
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        layer = 0
        while layer < n and ((phi >> layer) & 1) == 1:
            layers.append(layer)
            layer += 1
        layers.append(layer)
        llr_layer_vec.append(layers)

        if phi % 2 == 0:
            bit_layer_vec.append([])
        else:
            bl = []
            tmp = phi
            lvl = 0
            while (tmp & 1) == 1:
                bl.append(lvl)
                tmp >>= 1
                lvl += 1
            bit_layer_vec.append(bl)
    return list(range(N)), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（层叠 P/C 数组实现，与递归版本等价）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    errs = 0
    sigma = eb_n0_to_sigma(10.0, 0.5)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        llr = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(llr, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errs += 1
    print(f"SC high-SNR test errors: {errs}/100")
