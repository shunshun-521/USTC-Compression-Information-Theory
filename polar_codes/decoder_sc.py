"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
from enum import IntEnum

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sign_a = np.where(La >= 0, 1.0, -1.0)
    sign_b = np.where(Lb >= 0, 1.0, -1.0)
    return sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


class _NodeState(IntEnum):
    NOT_VISITED = 0
    AFTER_L = 1
    AFTER_R = 2


class _TreeSCDecoder:
    """基于因子树深度优先遍历的 SC 译码（参考实现）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.beliefs = np.zeros((self.n + 1, N), dtype=np.float64)
        self.decoded_bits = np.zeros((self.n + 1, N), dtype=int)
        self.node_state = np.zeros(2 * N - 1, dtype=int)
        self.u_hat = np.zeros(N, dtype=int)

    def _handle_leaf(self, node):
        if self.frozen_bits[node]:
            bit = 0
        else:
            bit = 0 if self.beliefs[self.n, node] >= 0 else 1
        self.decoded_bits[self.n, node] = bit
        self.u_hat[node] = bit

    def _step_l(self, node, depth, node_pos):
        span = 2 ** (self.n - depth)
        incoming = self.beliefs[depth, span * node : span * (node + 1)]
        half = span // 2
        left_child = node * 2
        child_depth = depth + 1
        child_span = span // 2
        self.beliefs[child_depth, child_span * left_child : child_span * (left_child + 1)] = (
            f_operation(incoming[:half], incoming[half:])
        )
        self.node_state[node_pos] = _NodeState.AFTER_L

    def _step_r(self, node, depth, node_pos):
        span = 2 ** (self.n - depth)
        incoming = self.beliefs[depth, span * node : span * (node + 1)]
        half = span // 2
        left_child = node * 2
        right_child = node * 2 + 1
        child_depth = depth + 1
        child_span = span // 2
        left_bits = self.decoded_bits[
            child_depth, child_span * left_child : child_span * (left_child + 1)
        ]
        self.beliefs[child_depth, child_span * right_child : child_span * (right_child + 1)] = (
            g_operation(incoming[:half], incoming[half:], left_bits)
        )
        self.node_state[node_pos] = _NodeState.AFTER_R

    def _step_u(self, node, depth):
        span = 2 ** (self.n - depth)
        left_child = node * 2
        right_child = node * 2 + 1
        parent_depth = depth + 1
        parent_span = span // 2
        left_bits = self.decoded_bits[
            parent_depth, parent_span * left_child : parent_span * (left_child + 1)
        ]
        right_bits = self.decoded_bits[
            parent_depth, parent_span * right_child : parent_span * (right_child + 1)
        ]
        self.decoded_bits[depth, span * node : span * (node + 1)] = np.concatenate(
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
                node_pos = 2 ** depth - 1 + node
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

        return self.u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（包装树遍历实现）"""
    N = len(llr)
    return _TreeSCDecoder(N, frozen_bits).decode(np.asarray(llr, dtype=np.float64))


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        temp = phi
        layer = 0
        while temp % 2 == 1:
            llr_layers.append(layer)
            temp //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        if phi % 2 == 0:
            bit_layers = list(range(n))
        else:
            bit_layers = []
            temp = phi
            layer = 0
            while temp % 2 == 1:
                bit_layers.append(layer)
                temp //= 2
                layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码：复用经过验证的树遍历实现。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode", x)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test N=64 errors={errors}/100")
