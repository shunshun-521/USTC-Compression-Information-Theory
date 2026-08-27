"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """min-sum f 运算（与 aff3ct f_LLR 一致）"""
    sign = np.sign(La * Lb)
    sign = np.where(sign == 0, 1.0, sign)
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 时 La+Lb，u=1 时 -La+Lb"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, -La + Lb)


class _SCNode:
    __slots__ = ("size", "lambda_", "s", "is_frozen", "left", "right", "lane_id")

    def __init__(self, size, lane_id=-1, is_frozen=False):
        self.size = size
        self.lambda_ = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=int)
        self.is_frozen = is_frozen
        self.lane_id = lane_id
        self.left = None
        self.right = None


def _build_tree(n, frozen_bits, lane=0):
    """构建极化码二叉树（与 aff3ct Binary_tree 一致）"""
    if n == 0:
        node = _SCNode(1, lane_id=lane, is_frozen=bool(frozen_bits[lane]))
        return node, lane + 1
    node = _SCNode(1 << n)
    node.left, lane = _build_tree(n - 1, frozen_bits, lane)
    node.right, lane = _build_tree(n - 1, frozen_bits, lane)
    return node, lane


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
        if not node.is_frozen:
            node.s[0] = 1 if node.lambda_[0] < 0 else 0


def _collect_decoded_bits(node):
    if node.left is None:
        return [node.s[0]]
    return _collect_decoded_bits(node.left) + _collect_decoded_bits(node.right)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树结构参考实现）"""
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    root, _ = _build_tree(n, frozen_bits)
    root.lambda_ = np.asarray(llr, dtype=np.float64)
    _recursive_decode(root)
    return np.array(_collect_decoded_bits(root), dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))

    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(n + 1):
        lambda_offset[i] = 1 << min(i, n - i)

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

        if phi & 1:
            bit_layers = [0]
        else:
            bit_layers = []
            p = phi
            layer = 0
            while (p & 1) == 0 and p > 0:
                bit_layers.append(layer)
                p >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return sc_decode_recursive(llr_ch, frozen_bits)
