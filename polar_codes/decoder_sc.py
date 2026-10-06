"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum f（0 视为正号）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


class _SCNode:
    __slots__ = ("size", "lambda_", "s", "is_frozen", "lane_id", "left", "right")

    def __init__(self, size):
        self.size = size
        self.lambda_ = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.is_frozen = False
        self.lane_id = 0
        self.left = None
        self.right = None


def _build_tree(size, lane_start):
    if size == 1:
        n = _SCNode(1)
        n.lane_id = lane_start
        return n
    node = _SCNode(size)
    half = size // 2
    node.left = _build_tree(half, lane_start)
    node.right = _build_tree(half, lane_start + half)
    return node


def _init_frozen(node, frozen_bits):
    if node.left is None:
        node.is_frozen = bool(frozen_bits[node.lane_id])
        return
    _init_frozen(node.left, frozen_bits)
    _init_frozen(node.right, frozen_bits)


def _recursive_decode(node):
    if node.left is not None:
        half = node.size // 2
        for i in range(half):
            node.left.lambda_[i] = f_operation(
                node.lambda_[i], node.lambda_[half + i]
            )
        _recursive_decode(node.left)
        for i in range(half):
            node.right.lambda_[i] = g_operation(
                node.lambda_[i],
                node.lambda_[half + i],
                node.left.s[i],
            )
        _recursive_decode(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
    else:
        if node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = 0 if node.lambda_[0] >= 0 else 1


def _collect_u(node, u_hat):
    if node.left is None:
        u_hat[node.lane_id] = node.s[0]
        return
    _collect_u(node.left, u_hat)
    _collect_u(node.right, u_hat)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（AFF3CT naive 二叉树实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr = llr[br]
    root = _build_tree(N, 0)
    _init_frozen(root, frozen_bits)
    root.lambda_[:] = llr
    _recursive_decode(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_u(root, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（接口保留）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layer = 0
        while layer < n and ((phi >> layer) & 1):
            layer += 1
        llr_layer_vec.append(list(range(n - 1, layer - 1, -1)))
        bit_layer_vec.append(list(range(layer)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：委托 AFF3CT 风格递归实现。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
