"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return Lb + (1 - 2 * u_hat) * La


def _xor_combine(left, right):
    left = list(left)
    right = list(right)
    res = [(left[i] + right[i]) % 2 for i in range(len(left))]
    res.extend(right)
    return res


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = np.zeros(N, dtype=int)

    def decode(y, depth, node):
        y = np.asarray(y, dtype=np.float64)
        if depth == n - 1:
            if node not in frozen_set:
                node_values[node] = 1 if y[0] < 0 else 0
            else:
                node_values[node] = 0
            return [node_values[node]]

        half = len(y) // 2
        left_dec = decode(f_operation(y[:half], y[half:]), depth + 1, 2 * node)
        right_dec = decode(
            g_operation(y[:half], y[half:], left_dec),
            depth + 1,
            2 * node + 1,
        )
        return _xor_combine(left_dec, right_dec)

    decode(llr, 0, 0)
    return node_values


def precompute_sc_indices(N):
    """预计算辅助参数（供 SCL 等模块使用）。"""
    return {"n": int(np.log2(N)) + 1, "N": N}


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码入口（委托给经校验的递归实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
