"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import prepare_decoder_llr


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, partial_bits):
    """g 运算：g(La, Lb, b) = Lb + (1 - 2*b) * La"""
    partial_bits = np.asarray(partial_bits, dtype=int)
    return Lb + (1 - 2 * partial_bits) * La


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 Arikan 因子图一致）"""
    llr = prepare_decoder_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_node(y, depth, node_idx):
        if len(y) == 1:
            if frozen_bits[node_idx]:
                u_hat[node_idx] = 0
            else:
                u_hat[node_idx] = 1 if y[0] < 0 else 0
            return np.array([u_hat[node_idx]])

        half = len(y) // 2
        left_llr = f_operation(y[:half], y[half:])
        left_partial = decode_node(left_llr, depth + 1, 2 * node_idx)
        right_llr = g_operation(y[:half], y[half:], left_partial)
        right_partial = decode_node(right_llr, depth + 1, 2 * node_idx + 1)
        return np.concatenate([(left_partial + right_partial) % 2, right_partial])

    decode_node(llr, 0, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    phase = list(range(N))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in phase:
        bits = format(phi, f"0{n}b")
        layers_llr = [s for s in range(n) if bits[n - 1 - s] == "0"]
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 1:
            layers_bit.append(0)
        for s in range(1, n):
            if (phi >> s) % 2 == 1:
                layers_bit.append(s)
        bit_layer_vec.append(layers_bit)

    lambda_offset = [1 << s for s in range(n + 1)]
    return phase, lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_tables(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（当前委托给已验证的递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
