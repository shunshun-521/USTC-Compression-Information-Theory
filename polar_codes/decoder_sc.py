"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
from collections import namedtuple

import numpy as np
from encoder import bit_reversal_permutation

_Frame = namedtuple('_Frame', 'y depth node phase left_bits L1 L2')


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    if np.isscalar(La) and np.isscalar(Lb):
        return np.sign(La) * np.sign(Lb) * min(abs(La), abs(Lb))
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    if np.isscalar(La):
        return Lb + (1 - 2 * u_hat) * La
    return Lb + (1 - 2 * u_hat) * La


def _vector_f(L1, L2):
    return [f_operation(a, b) for a, b in zip(L1, L2)]


def _vector_g(L1, L2, bits):
    return [g_operation(a, b, u) for a, b, u in zip(L1, L2, bits)]


def _xor_merge(left, right):
    return [(a + b) % 2 for a, b in zip(left, right)] + list(right)


def _prepare_llr(llr_ch, N):
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def _decode_tree(llr, frozen_set, N, n, node_values, use_stack):
    if use_stack:
        stack = [_Frame(list(llr), 0, 0, 'enter', None, None, None)]
        returns = []
        while stack:
            frame = stack[-1]
            if frame.phase == 'enter':
                if frame.depth == n - 1:
                    stack.pop()
                    bit = 0 if frame.node in frozen_set or frame.y[0] >= 0 else 1
                    node_values[frame.node] = bit
                    returns.append([bit])
                else:
                    half = len(frame.y) // 2
                    L1, L2 = frame.y[:half], frame.y[half:]
                    left_llr = _vector_f(L1, L2)
                    stack[-1] = _Frame(frame.y, frame.depth, frame.node, 'after_left', None, L1, L2)
                    stack.append(_Frame(left_llr, frame.depth + 1, 2 * frame.node, 'enter', None, None, None))
            elif frame.phase == 'after_left':
                left_bits = returns.pop()
                right_llr = _vector_g(frame.L1, frame.L2, left_bits)
                stack[-1] = _Frame(frame.y, frame.depth, frame.node, 'after_right', left_bits, frame.L1, frame.L2)
                stack.append(_Frame(right_llr, frame.depth + 1, 2 * frame.node + 1, 'enter', None, None, None))
            else:
                right_bits = returns.pop()
                stack.pop()
                returns.append(_xor_merge(frame.left_bits, right_bits))
        return

    def decode_node(y, depth, node):
        if depth == n - 1:
            bit = 0 if node in frozen_set or y[0] >= 0 else 1
            node_values[node] = bit
            return [bit]

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_bits = decode_node(_vector_f(L1, L2), depth + 1, 2 * node)
        right_bits = decode_node(_vector_g(L1, L2, left_bits), depth + 1, 2 * node + 1)
        return _xor_merge(left_bits, right_bits)

    decode_node(list(llr), 0, 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _prepare_llr(llr, len(llr))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = [0] * N
    _decode_tree(llr, frozen_set, N, n, node_values, use_stack=False)
    return np.array(node_values, dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（兼容接口）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, list(range(N)), list(range(n))


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（显式栈实现）。"""
    llr = _prepare_llr(llr_ch, len(llr_ch))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    node_values = [0] * N
    _decode_tree(llr, frozen_set, N, n, node_values, use_stack=True)
    return np.array(node_values, dtype=int)
