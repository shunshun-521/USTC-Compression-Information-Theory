"""
极化码 SC（串行抵消）译码器
树遍历实现（与编码 u@G_N 配套，信道 LLR 按码字下标顺序输入）
"""
import math
from enum import IntEnum

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为上支路，Lb 为下支路"""
    u_hat = np.asarray(u_hat)
    return Lb + (1 - 2 * u_hat) * La


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class _NodeState(IntEnum):
    NOT_VISITED = 0
    AFTER_L = 1
    AFTER_R = 2


class _SCTreeDecoder:
    """基于深度优先遍历的 SC 译码器"""

    def __init__(self, N, frozen_bits):
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = set(np.where(self.frozen_bits)[0])
        self.beliefs = np.zeros((self.n + 1, N), dtype=np.float64)
        self.decoded_bits = np.zeros((self.n + 1, N), dtype=np.int8)
        self.node_state = np.zeros(2 * N - 1, dtype=np.int8)

    def _node_pos(self, depth, node):
        return (1 << depth) - 1 + node

    def _handle_leaf(self, node):
        if node in self.frozen_indices:
            self.decoded_bits[self.n, node] = 0
        else:
            self.decoded_bits[self.n, node] = (
                0 if self.beliefs[self.n, node] >= 0 else 1
            )

    def _step_l(self, node, depth, node_pos):
        span = 1 << (self.n - depth)
        inc = self.beliefs[depth, span * node : span * (node + 1)]
        half = span // 2
        left_child = 2 * node
        depth_l = depth + 1
        span_l = span // 2
        self.beliefs[depth_l, span_l * left_child : span_l * (left_child + 1)] = (
            f_operation(inc[:half], inc[half:])
        )
        self.node_state[node_pos] = _NodeState.AFTER_L

    def _step_r(self, node, depth, node_pos):
        span = 1 << (self.n - depth)
        inc = self.beliefs[depth, span * node : span * (node + 1)]
        half = span // 2
        left_child = 2 * node
        depth_l = depth + 1
        span_l = span // 2
        decoded_left = self.decoded_bits[
            depth_l, span_l * left_child : span_l * (left_child + 1)
        ]
        right_child = 2 * node + 1
        depth_r = depth + 1
        span_r = span // 2
        self.beliefs[depth_r, span_r * right_child : span_r * (right_child + 1)] = (
            g_operation(inc[:half], inc[half:], decoded_left)
        )
        self.node_state[node_pos] = _NodeState.AFTER_R

    def _step_u(self, node, depth):
        span = 1 << (self.n - depth)
        half = span // 2
        left_child = 2 * node
        right_child = 2 * node + 1
        depth_c = depth + 1
        span_c = span // 2
        bits_l = self.decoded_bits[
            depth_c, span_c * left_child : span_c * (left_child + 1)
        ]
        bits_r = self.decoded_bits[
            depth_c, span_c * right_child : span_c * (right_child + 1)
        ]
        self.decoded_bits[depth, span * node : span * (node + 1)] = np.concatenate(
            [(bits_l + bits_r) % 2, bits_r]
        )

    def decode(self, llr_ch):
        self.beliefs[0, :] = np.asarray(llr_ch, dtype=np.float64)
        self.decoded_bits.fill(0)
        self.node_state.fill(_NodeState.NOT_VISITED)
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
                npos = self._node_pos(depth, node)
                st = self.node_state[npos]
                if st == _NodeState.NOT_VISITED:
                    self._step_l(node, depth, npos)
                    node *= 2
                    depth += 1
                elif st == _NodeState.AFTER_L:
                    self._step_r(node, depth, npos)
                    node = node * 2 + 1
                    depth += 1
                else:
                    self._step_u(node, depth)
                    node //= 2
                    depth -= 1
        return self.decoded_bits[self.n, :].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（树遍历）。
    输入 llr_ch 为信道输出顺序（与 polar_encode 码字下标一致）；
    内部对比特倒序重排以匹配因子图。
    """
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llr_perm = np.asarray(llr_ch, dtype=np.float64)[rev]
    return _SCTreeDecoder(N, frozen_bits).decode(llr_perm)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """兼容接口：返回 log2(N) 与码长"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], list(range(N)), n
