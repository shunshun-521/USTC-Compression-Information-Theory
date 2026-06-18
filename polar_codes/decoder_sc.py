"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
from enum import IntEnum

import numpy as np

from encoder import bit_reversal_permutation


class _NodeState(IntEnum):
    NOT_VISITED = 0
    AFTER_L = 1
    AFTER_R = 2


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode(llr_vec, idx_range):
        n = len(llr_vec)
        if n == 1:
            i = idx_range[0]
            u_hat[i] = 0 if frozen_bits[i] or llr_vec[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_vec[:half], llr_vec[half:])
        decode(llr_left, idx_range[:half])
        llr_right = g_operation(llr_vec[:half], llr_vec[half:], u_hat[idx_range[:half]])
        decode(llr_right, idx_range[half:])

    decode(llr, np.arange(N))
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        for b in range(n):
            if ((phi >> b) & 1) == 0:
                layers = list(range(b, n))
                break
        else:
            layers = []
        llr_layer_vec.append(layers)

        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layer_vec.append(list(range(n - 1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCTreeDecoder:
    """树遍历 SC 译码内核（非递归高效实现）。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.beliefs = np.zeros((self.n + 1, N), dtype=np.float64)
        self.decoded_bits = np.zeros((self.n + 1, N), dtype=int)
        self.node_state = np.zeros(2 * N - 1, dtype=int)

    def _node_pos(self, depth, node):
        return (1 << depth) - 1 + node

    def _handle_leaf(self, node):
        if self.frozen_bits[node]:
            self.decoded_bits[self.n, node] = 0
        else:
            self.decoded_bits[self.n, node] = (
                0 if self.beliefs[self.n, node] >= 0 else 1
            )

    def _step_l(self, node, depth):
        span = 1 << (self.n - depth)
        start = span * node
        end = start + span
        incoming = self.beliefs[depth, start:end]
        left = incoming[: span // 2]
        right = incoming[span // 2 :]
        child = node * 2
        cdepth = depth + 1
        cspan = span // 2
        cstart = cspan * child
        self.beliefs[cdepth, cstart : cstart + cspan] = f_operation(left, right)
        self.node_state[self._node_pos(depth, node)] = _NodeState.AFTER_L

    def _step_r(self, node, depth):
        span = 1 << (self.n - depth)
        start = span * node
        end = start + span
        incoming = self.beliefs[depth, start:end]
        left = incoming[: span // 2]
        right = incoming[span // 2 :]

        left_child = node * 2
        lcdepth = depth + 1
        lcspan = span // 2
        lcstart = lcspan * left_child
        u_left = self.decoded_bits[lcdepth, lcstart : lcstart + lcspan]

        child = node * 2 + 1
        cdepth = depth + 1
        cspan = span // 2
        cstart = cspan * child
        self.beliefs[cdepth, cstart : cstart + cspan] = g_operation(left, right, u_left)
        self.node_state[self._node_pos(depth, node)] = _NodeState.AFTER_R

    def _step_u(self, node, depth):
        span = 1 << (self.n - depth)
        start = span * node
        pdepth = depth + 1
        pspan = span // 2
        left_child = node * 2
        right_child = node * 2 + 1
        lstart = pspan * left_child
        rstart = pspan * right_child
        left_bits = self.decoded_bits[pdepth, lstart : lstart + pspan]
        right_bits = self.decoded_bits[pdepth, rstart : rstart + pspan]
        self.decoded_bits[depth, start : start + span] = np.concatenate(
            [(left_bits + right_bits) % 2, right_bits]
        )

    def decode(self, llr_ch):
        self.beliefs.fill(0.0)
        self.decoded_bits.fill(0)
        self.node_state.fill(0)
        self.beliefs[0, :] = np.asarray(llr_ch, dtype=np.float64)

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
                pos = self._node_pos(depth, node)
                state = self.node_state[pos]
                if state == _NodeState.NOT_VISITED:
                    self._step_l(node, depth)
                    node = node * 2
                    depth += 1
                elif state == _NodeState.AFTER_L:
                    self._step_r(node, depth)
                    node = node * 2 + 1
                    depth += 1
                else:
                    self._step_u(node, depth)
                    node //= 2
                    depth -= 1

        return self.decoded_bits[self.n, :].copy()


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（树遍历实现）。
    """
    N = len(llr_ch)
    decoder = _SCTreeDecoder(N, frozen_bits)
    return decoder.decode(llr_ch)


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode x =", x)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(12.0, K / N)
    errors = 0
    for _ in range(100):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_src)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_rec[info_idx], u_src[info_idx]):
            errors += 1
    print(f"SC low-noise test: {errors}/100 frame errors")
    assert errors == 0
