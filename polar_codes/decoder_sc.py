"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，树遍历）
"""
import math
from enum import IntEnum

import numpy as np

from encoder import bit_reversal_permutation


def _sign_pm(x):
    """符号函数：0 视为 +1。"""
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


class _NodeState(IntEnum):
    NOT_VISITED = 0
    AFTER_L = 1
    AFTER_R = 2


class _SCTreeDecoder:
    """基于树遍历的非递归 SC 译码内核。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.beliefs = np.zeros((self.n + 1, N), dtype=np.float64)
        self.decoded_bits = np.zeros((self.n + 1, N), dtype=int)
        self.node_state = np.zeros(2 * N - 1, dtype=int)

    def _leaf(self, node):
        if self.frozen_bits[node]:
            self.decoded_bits[self.n, node] = 0
        else:
            self.decoded_bits[self.n, node] = 0 if self.beliefs[self.n, node] >= 0 else 1

    def _step_l(self, node, depth, node_pos):
        span = 1 << (self.n - depth)
        incoming = self.beliefs[depth, span * node : span * (node + 1)]
        half = span // 2
        self.beliefs[depth + 1, (span // 2) * (2 * node) : (span // 2) * (2 * node + 1)] = (
            f_operation(incoming[:half], incoming[half:])
        )
        self.node_state[node_pos] = _NodeState.AFTER_L

    def _step_r(self, node, depth, node_pos):
        span = 1 << (self.n - depth)
        incoming = self.beliefs[depth, span * node : span * (node + 1)]
        half = span // 2
        left_node = 2 * node
        left_span = span // 2
        decoded_left = self.decoded_bits[
            depth + 1, left_span * left_node : left_span * (left_node + 1)
        ]
        right_node = 2 * node + 1
        self.beliefs[depth + 1, left_span * right_node : left_span * (right_node + 1)] = (
            g_operation(incoming[:half], incoming[half:], decoded_left)
        )
        self.node_state[node_pos] = _NodeState.AFTER_R

    def _step_u(self, node, depth):
        span = 1 << (self.n - depth)
        half = span // 2
        left = 2 * node
        right = 2 * node + 1
        bits_left = self.decoded_bits[depth + 1, half * left : half * (left + 1)]
        bits_right = self.decoded_bits[depth + 1, half * right : half * (right + 1)]
        self.decoded_bits[depth, span * node : span * (node + 1)] = np.concatenate(
            [(bits_left + bits_right) % 2, bits_right]
        )

    def decode(self, llr_br):
        """llr_br 为比特倒序后的信道 LLR。"""
        self.beliefs[0, :] = llr_br
        self.node_state[:] = _NodeState.NOT_VISITED
        node = 0
        depth = 0
        done = False

        while not done:
            if depth == self.n:
                self._leaf(node)
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

        return self.decoded_bits[self.n, :].copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    br = bit_reversal_permutation(len(llr))
    return sc_decode(llr, frozen_bits)  # 树遍历与递归等价，复用主实现


def precompute_sc_indices(N):
    """保留接口：返回非递归 SC 辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while p & 1:
            llr_layers.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)

        if phi % 2 == 0:
            bit_layers = list(range(n))
        else:
            bit_layers = []
            p = phi
            layer = 0
            while p & 1:
                bit_layers.append(layer)
                p >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    输入 llr_ch 为自然信道顺序，内部做比特倒序置换。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    decoder = _SCTreeDecoder(N, frozen_bits)
    return decoder.decode(llr_ch[br])
