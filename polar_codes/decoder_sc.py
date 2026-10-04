"""
极化码 SC（串行抵消）译码器
"""
import numpy as np


def f_operation(La, Lb):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.int64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_decode_core(llr, frozen_set, n_depth):
    llr = np.asarray(llr, dtype=np.float64).tolist()
    N = len(llr)
    node_values = [0] * N

    def decode(y, depth, node):
        if depth == n_depth - 1:
            if node in frozen_set:
                node_values[node] = 0
            else:
                node_values[node] = 1 if y[0] < 0 else 0
            return [node_values[node]]

        half = len(y) // 2
        l1 = y[:half]
        l2 = y[half:]
        left = [
            np.sign(a) * np.sign(b) * min(abs(a), abs(b)) for a, b in zip(l1, l2)
        ]
        arr1 = decode(left, depth + 1, 2 * node)
        right = [l2[i] + (1 - 2 * arr1[i]) * l1[i] for i in range(len(l1))]
        arr2 = decode(right, depth + 1, 2 * node + 1)
        return arr1 + arr2

    decode(llr, 0, 0)
    return np.array(node_values, dtype=int)


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n_depth = int(np.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    return _sc_decode_core(llr_ch, frozen_set, n_depth)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 2)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]
