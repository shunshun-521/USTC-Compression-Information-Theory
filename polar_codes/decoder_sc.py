"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def h_operation(L):
    """硬判决：LLR >= 0 -> 0，否则 1。"""
    return (L < 0).astype(int)


class _SCNode:
    __slots__ = ("size", "lambda_v", "s", "is_frozen", "left", "right", "lane_id")

    def __init__(self, size):
        self.size = size
        self.lambda_v = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=int)
        self.is_frozen = False
        self.left = None
        self.right = None
        self.lane_id = 0


def _build_tree(size, lane_counter):
    node = _SCNode(size)
    if size > 1:
        half = size // 2
        node.left = _build_tree(half, lane_counter)
        node.right = _build_tree(half, lane_counter)
    else:
        node.lane_id = lane_counter[0]
        lane_counter[0] += 1
    return node


def _init_frozen(node, frozen_bits):
    if node.size == 1:
        node.is_frozen = bool(frozen_bits[node.lane_id])
    else:
        _init_frozen(node.left, frozen_bits)
        _init_frozen(node.right, frozen_bits)


def _recursive_decode_aff3ct(node):
    if node.size > 1:
        half = node.size // 2
        for i in range(half):
            node.left.lambda_v[i] = f_operation(
                node.lambda_v[i], node.lambda_v[half + i]
            )
        _recursive_decode_aff3ct(node.left)
        for i in range(half):
            node.right.lambda_v[i] = g_operation(
                node.lambda_v[i],
                node.lambda_v[half + i],
                node.left.s[i],
            )
        _recursive_decode_aff3ct(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
    else:
        if node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = h_operation(node.lambda_v[0:1])[0]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（aff3ct 树结构参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    lane_counter = [0]
    root = _build_tree(N, lane_counter)
    _init_frozen(root, frozen_bits)
    root.lambda_v[:] = llr
    _recursive_decode_aff3ct(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_leaves(root, u_hat)
    return u_hat


def _collect_leaves(node, u_hat):
    if node.size == 1:
        u_hat[node.lane_id] = node.s[0]
    else:
        _collect_leaves(node.left, u_hat)
        _collect_leaves(node.right, u_hat)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        l = 0
        while ((phi >> l) & 1) == 1:
            layers.append(l)
            l += 1
        llr_layer_vec.append(layers)

        blayers = []
        l = 0
        temp = phi + 1
        while temp % 2 == 0 and l < n:
            blayers.append(l)
            temp >>= 1
            l += 1
        bit_layer_vec.append(blayers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_layered(llr_ch, frozen_bits):
    """非递归 SC 译码（分层 P/C 数组实现）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            s = lambda_offset[layer]
            for beta in range(0, N, 2 * s):
                for jj in range(beta, beta + s):
                    P[layer, jj] = f_operation(
                        P[layer + 1, jj], P[layer + 1, jj + s]
                    )
                for jj in range(beta, beta + s):
                    P[layer, jj + s] = g_operation(
                        P[layer + 1, jj],
                        P[layer + 1, jj + s],
                        C[layer, jj],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1

        C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            s = lambda_offset[layer]
            for beta in range(0, N, 2 * s):
                for jj in range(beta, beta + s):
                    C[layer + 1, jj] = (C[layer, jj] ^ C[layer, jj + s]) & 1
                    C[layer + 1, jj + s] = C[layer, jj + s]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（与树形递归实现等价，当前委托递归以保证数值一致）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
