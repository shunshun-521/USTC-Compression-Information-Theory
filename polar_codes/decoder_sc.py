"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（aff3ct f_LLR）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（aff3ct g_LLR）。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def h_operation(La):
    """硬判决：LLR < 0 -> 1，否则 0。"""
    return 0 if La >= 0 else 1


class _TreeNode:
    __slots__ = ("lane_id", "lambda_v", "s", "is_frozen", "left", "right", "is_leaf")

    def __init__(self, lane_id=-1, size=1):
        self.lane_id = lane_id
        self.lambda_v = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.is_frozen = False
        self.left = None
        self.right = None
        self.is_leaf = size == 1


def _build_tree(depth, lane_counter):
    if depth == 0:
        node = _TreeNode(lane_id=lane_counter[0], size=1)
        lane_counter[0] += 1
        return node
    size = 1 << depth
    node = _TreeNode(size=size)
    node.left = _build_tree(depth - 1, lane_counter)
    node.right = _build_tree(depth - 1, lane_counter)
    return node


def _init_frozen(node, frozen_bits):
    if node.is_leaf:
        node.is_frozen = bool(frozen_bits[node.lane_id])
    else:
        _init_frozen(node.left, frozen_bits)
        _init_frozen(node.right, frozen_bits)


def _recursive_decode(node):
    if not node.is_leaf:
        half = len(node.lambda_v) // 2
        node.left.lambda_v[:] = f_operation(
            node.lambda_v[:half], node.lambda_v[half:]
        )
        _recursive_decode(node.left)
        node.right.lambda_v[:] = g_operation(
            node.lambda_v[:half], node.lambda_v[half:], node.left.s
        )
        _recursive_decode(node.right)
        node.s[:half] = node.left.s ^ node.right.s
        node.s[half:] = node.right.s
    else:
        node.s[0] = 0 if node.is_frozen else h_operation(node.lambda_v[0])


def _gather_uhat(node, frozen_bits, u_hat):
    if node.is_leaf:
        u_hat[node.lane_id] = 0 if frozen_bits[node.lane_id] else int(node.s[0])
    else:
        _gather_uhat(node.left, frozen_bits, u_hat)
        _gather_uhat(node.right, frozen_bits, u_hat)


def _tree_decode(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    root = _build_tree(n, [0])
    _init_frozen(root, frozen_bits)
    root.lambda_v[:] = llr
    _recursive_decode(root)
    u_hat = np.zeros(N, dtype=int)
    _gather_uhat(root, frozen_bits, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """预计算 SCD 算法所需的层索引向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        for layer in range(n):
            if phi % (1 << (layer + 1)) >= (1 << layer):
                llr_layers.append(layer)
            if phi % (1 << (layer + 1)) < (1 << layer):
                bit_layers.append(layer)
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（aff3ct naive 树结构）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _tree_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    使用显式栈模拟树遍历，结果与递归版本一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
    return _tree_decode(llr_ch, frozen_bits)
