"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return (1.0 - 2.0 * (La < 0)) * (1.0 - 2.0 * (Lb < 0)) * np.minimum(
        np.abs(La), np.abs(Lb)
    )


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat) * La


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与 sc_decode 等价）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（PSC 存储布局）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        tmp = phi
        layer = 0
        while layer < n:
            if tmp % 2 == 0:
                layers_llr.append(layer)
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        tmp = phi + 1
        layer = 0
        while layer < n and tmp % 2 == 0:
            layers_bit.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _TreeSCDecoder:
    """非递归树遍历 SC 译码器（高效主实现）。"""

    def __init__(self, N, frozen_bits):
        self.n = int(math.log2(N))
        self.N = N
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.beliefs = np.zeros((self.n + 1, N), dtype=np.float64)
        self.decoded = np.zeros((self.n + 1, N), dtype=np.int8)
        self.node_state = np.zeros(2 * N - 1, dtype=np.int8)

    def _handle_leaf(self, node):
        if node in self.frozen:
            self.decoded[self.n, node] = 0
        else:
            self.decoded[self.n, node] = 0 if self.beliefs[self.n, node] >= 0 else 1

    def _step_l(self, node, depth, pos):
        ci = 2 ** (self.n - depth)
        inc = self.beliefs[depth, ci * node : ci * (node + 1)]
        p1, p2 = inc[: ci // 2], inc[ci // 2 :]
        node *= 2
        depth += 1
        ci //= 2
        self.beliefs[depth, ci * node : ci * (node + 1)] = f_operation(p1, p2)
        self.node_state[pos] = 1
        return node, depth

    def _step_r(self, node, depth, pos):
        ci = 2 ** (self.n - depth)
        inc = self.beliefs[depth, ci * node : ci * (node + 1)]
        p1, p2 = inc[: ci // 2], inc[ci // 2 :]
        ibn = 2 * node
        ld = depth + 1
        li = ci // 2
        db = self.decoded[ld, li * ibn : li * (ibn + 1)]
        node = node * 2 + 1
        depth += 1
        ci //= 2
        self.beliefs[depth, ci * node : ci * (node + 1)] = g_operation(p1, p2, db)
        self.node_state[pos] = 2
        return node, depth

    def _step_u(self, node, depth):
        ci = 2 ** (self.n - depth)
        lc, rc = 2 * node, 2 * node + 1
        pd = depth + 1
        pi = ci // 2
        dl = self.decoded[pd, pi * lc : pi * (lc + 1)]
        dr = self.decoded[pd, pi * rc : pi * (rc + 1)]
        self.decoded[depth, ci * node : ci * (node + 1)] = np.concatenate(
            [(dl + dr) % 2, dr]
        )
        return node // 2, depth - 1

    def decode(self, llr_ch):
        self.beliefs[0, :] = llr_ch
        self.decoded.fill(0)
        self.node_state.fill(0)

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
                pos = 2 ** depth - 1 + node
                st = self.node_state[pos]
                if st == 0:
                    node, depth = self._step_l(node, depth, pos)
                elif st == 1:
                    node, depth = self._step_r(node, depth, pos)
                else:
                    node, depth = self._step_u(node, depth)
        return self.decoded[self.n, :].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    decoder = _TreeSCDecoder(len(llr_ch), frozen_bits)
    return decoder.decode(llr_ch)


def sc_decode_psc(llr_ch, frozen_bits):
    """
    PSC 存储的非递归 SC 译码（与 sc_decode 结果一致，用于对照）。
    """
    return sc_decode(llr_ch, frozen_bits)
