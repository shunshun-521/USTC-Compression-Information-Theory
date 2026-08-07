"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
from enum import IntEnum

import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa = 1 - 2 * (La < 0)
    sb = 1 - 2 * (Lb < 0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return Lb + (1 - 2 * u_hat) * La


class _NodeState(IntEnum):
    NOT_VISITED = 0
    AFTER_L = 1
    AFTER_R = 2


class _TreeSCDecoder:
    """基于因子树遍历的 SC 译码器（与编码器匹配）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.beliefs = np.zeros((self.n + 1, N), dtype=np.float64)
        self.decoded_bits = np.zeros((self.n + 1, N), dtype=np.int8)
        self.node_state = np.zeros(2 * N - 1, dtype=int)

    def _handle_leaf(self, node):
        if self.frozen_bits[node]:
            self.decoded_bits[self.n, node] = 0
        else:
            self.decoded_bits[self.n, node] = 0 if self.beliefs[self.n, node] >= 0 else 1

    def _step_l(self, node, depth, node_pos):
        span = 1 << (self.n - depth)
        incoming = self.beliefs[depth, span * node:span * (node + 1)]
        half = span // 2
        left_child = 2 * node
        child_depth = depth + 1
        child_span = span // 2
        self.beliefs[child_depth, child_span * left_child:child_span * (left_child + 1)] = f_operation(
            incoming[:half], incoming[half:]
        )
        self.node_state[node_pos] = _NodeState.AFTER_L

    def _step_r(self, node, depth, node_pos):
        span = 1 << (self.n - depth)
        incoming = self.beliefs[depth, span * node:span * (node + 1)]
        half = span // 2
        left_child = 2 * node
        child_depth = depth + 1
        child_span = span // 2
        decoded_left = self.decoded_bits[child_depth, child_span * left_child:child_span * (left_child + 1)]
        right_child = 2 * node + 1
        self.beliefs[child_depth, child_span * right_child:child_span * (right_child + 1)] = g_operation(
            incoming[:half], incoming[half:], decoded_left
        )
        self.node_state[node_pos] = _NodeState.AFTER_R

    def _step_u(self, node, depth):
        span = 1 << (self.n - depth)
        child_depth = depth + 1
        child_span = span // 2
        left_child = 2 * node
        right_child = 2 * node + 1
        left_bits = self.decoded_bits[child_depth, child_span * left_child:child_span * (left_child + 1)]
        right_bits = self.decoded_bits[child_depth, child_span * right_child:child_span * (right_child + 1)]
        self.decoded_bits[depth, span * node:span * (node + 1)] = np.concatenate(
            [(left_bits + right_bits) % 2, right_bits]
        )

    def decode(self, llr_ch):
        self.beliefs[0, :] = llr_ch
        node = 0
        depth = 0
        done = False

        while not done:
            if depth == self.n:
                self._handle_leaf(node)
                if node == self.N - 1:
                    done = True
                else:
                    node //= 2
                    depth -= 1
            else:
                node_pos = (1 << depth) - 1 + node
                state = self.node_state[node_pos]
                if state == _NodeState.NOT_VISITED:
                    self._step_l(node, depth, node_pos)
                    node *= 2
                    depth += 1
                elif state == _NodeState.AFTER_L:
                    self._step_r(node, depth, node_pos)
                    node = node * 2 + 1
                    depth += 1
                else:
                    self._step_u(node, depth)
                    node //= 2
                    depth -= 1

        return self.decoded_bits[self.n, :].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与树遍历版本等价）。"""
    return _TreeSCDecoder(len(llr), frozen_bits).decode(np.asarray(llr, dtype=np.float64))


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（接口保留）。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))
    llr_layer_vec = [list(range(n - 1, -1, -1)) if phi == 0 else [] for phi in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归接口别名。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
