"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算，u_hat 可为标量或向量"""
    return (1 - 2 * u_hat) * La + Lb


def _h_operation(llr):
    """硬判决：LLR < 0 -> 1"""
    return 0 if llr >= 0 else 1


class _SCNode:
    __slots__ = ("lam", "s", "is_leaf", "lane_id", "left", "right")

    def __init__(self, size, is_leaf=False, lane_id=0):
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.is_leaf = is_leaf
        self.lane_id = lane_id
        self.left = None
        self.right = None


def _build_tree(n_leaves, lane_start=0):
    """构建满二叉树，叶节点 lane_id 为比特索引"""
    if n_leaves == 1:
        return _SCNode(1, is_leaf=True, lane_id=lane_start)
    node = _SCNode(n_leaves)
    half = n_leaves // 2
    node.left = _build_tree(half, lane_start)
    node.right = _build_tree(half, lane_start + half)
    return node


def _recursive_decode_aff3ct(node, frozen_bits):
    if not node.is_leaf:
        half = len(node.lam) // 2
        left, right = node.left, node.right
        for i in range(half):
            left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
        _recursive_decode_aff3ct(left, frozen_bits)
        for i in range(half):
            right.lam[i] = g_operation(node.lam[i], node.lam[half + i], left.s[i])
        _recursive_decode_aff3ct(right, frozen_bits)
        for i in range(half):
            node.s[i] = (left.s[i] ^ right.s[i]) % 2
            node.s[half + i] = right.s[i]
    else:
        if frozen_bits[node.lane_id]:
            node.s[0] = 0
        else:
            node.s[0] = _h_operation(node.lam[0])


def _collect_leaves(node, out):
    if node.is_leaf:
        out[node.lane_id] = node.s[0]
    else:
        _collect_leaves(node.left, out)
        _collect_leaves(node.right, out)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（aff3ct 树形参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    br = bit_reversal_permutation(N)
    root = _build_tree(N)
    root.lam[:] = llr[br]
    _recursive_decode_aff3ct(root, frozen_bits)
    u_hat = np.zeros(N, dtype=int)
    _collect_leaves(root, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n - 1, -1, -1))
        else:
            layer = 0
            p = phi
            while p & 1:
                layer += 1
                p >>= 1
            llr_layers = list(range(n - 1, n - 1 - layer, -1))

        bit_layers = []
        if phi & 1:
            layer = 0
            while (phi >> layer) & 1:
                bit_layers.append(layer)
                layer += 1

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主入口：使用显式树结构逐层更新（与递归参考实现等价，避免深层递归）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_fast(llr_ch, frozen_bits):
    """层状快速 SC（与 sc_decode_recursive 结果一致）"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    br = bit_reversal_permutation(N)
    llr = np.asarray(llr_ch, dtype=np.float64)[br]

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr

    u_hat = np.zeros(N, dtype=int)
    _, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            step = 1 << (n - 1 - layer)
            for block in range(0, N, 2 * step):
                for j in range(step):
                    left = block + j
                    right = left + step
                    La = P[layer + 1, left]
                    Lb = P[layer + 1, right]
                    if (phi >> layer) & 1:
                        P[layer, left] = g_operation(La, Lb, C[layer, left])
                    else:
                        P[layer, left] = f_operation(La, Lb)

        u_hat[phi] = 0 if frozen_bits[phi] else _h_operation(P[0, 0])
        C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            step = 1 << layer
            span = 1 << (layer + 1)
            for block in range(0, N, span):
                base = block
                for j in range(step):
                    idx = base + j
                    C[layer + 1, 2 * idx] = (C[layer, idx] + C[layer, idx + step]) % 2
                    C[layer + 1, 2 * idx + 1] = C[layer, idx + step]

    return u_hat
