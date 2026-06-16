"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
from enum import IntEnum

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


class _NodeState(IntEnum):
    NOT_VISITED = 0
    AFTER_L = 1
    AFTER_R = 2


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int32)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        p = phi
        for layer in range(n):
            if p % 2 == 0:
                llr_layer_vec[phi].append(layer)
                p //= 2
            else:
                break

        p = phi
        for layer in range(n):
            if p % 2 == 1:
                bit_layer_vec[phi].append(layer)
                p //= 2
            else:
                break

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（基于因子图 L/R/U 遍历，高效实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    beliefs = np.zeros((n + 1, N), dtype=np.float64)
    decoded = np.zeros((n + 1, N), dtype=np.int32)
    node_state = np.zeros(2 * N - 1, dtype=np.int32)
    beliefs[0, :] = llr_ch

    node = 0
    depth = 0
    done = False

    while not done:
        if depth == n:
            if frozen_bits[node]:
                decoded[n, node] = 0
            else:
                decoded[n, node] = 0 if beliefs[n, node] >= 0 else 1

            if node == N - 1:
                done = True
            else:
                node //= 2
                depth -= 1
        else:
            node_pos = (1 << depth) - 1 + node
            span = 1 << (n - depth)
            start = span * node
            end = start + span

            if node_state[node_pos] == _NodeState.NOT_VISITED:
                left = beliefs[depth, start : start + span // 2]
                right = beliefs[depth, start + span // 2 : end]
                child = node * 2
                child_depth = depth + 1
                child_span = span // 2
                child_start = child_span * child
                beliefs[child_depth, child_start : child_start + child_span] = f_operation(
                    left, right
                )
                node_state[node_pos] = _NodeState.AFTER_L
                node = child
                depth += 1

            elif node_state[node_pos] == _NodeState.AFTER_L:
                left = beliefs[depth, start : start + span // 2]
                right = beliefs[depth, start + span // 2 : end]
                left_child = node * 2
                left_depth = depth + 1
                left_span = span // 2
                left_start = left_span * left_child
                left_bits = decoded[left_depth, left_start : left_start + left_span]

                child = node * 2 + 1
                child_depth = depth + 1
                child_span = span // 2
                child_start = child_span * child
                beliefs[child_depth, child_start : child_start + child_span] = g_operation(
                    left, right, left_bits
                )
                node_state[node_pos] = _NodeState.AFTER_R
                node = child
                depth += 1

            else:
                left_child = node * 2
                right_child = node * 2 + 1
                parent_depth = depth + 1
                parent_span = span // 2
                left_start = parent_span * left_child
                right_start = parent_span * right_child
                left_bits = decoded[parent_depth, left_start : left_start + parent_span]
                right_bits = decoded[parent_depth, right_start : right_start + parent_span]
                decoded[depth, start:end] = np.concatenate(
                    [(left_bits + right_bits) % 2, right_bits]
                )
                node //= 2
                depth -= 1

    return decoded[n, :].astype(np.int32)
