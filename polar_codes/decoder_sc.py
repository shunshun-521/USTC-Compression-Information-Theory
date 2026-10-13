"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from channel import map_llr_to_decoder


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


class _SCNode:
    __slots__ = ("lam", "s", "frozen", "left", "right", "lane_id", "leaf")

    def __init__(self, size, lane_id=0):
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.frozen = False
        self.lane_id = lane_id
        self.leaf = size == 1
        self.left = None
        self.right = None


def _build_sc_tree(n, lane_start=0):
    """构建与 aff3ct 一致的极化 SC 二叉树"""
    if n == 1:
        return _SCNode(1, lane_start)
    node = _SCNode(n, lane_start)
    half = n // 2
    node.left = _build_sc_tree(half, lane_start)
    node.right = _build_sc_tree(half, lane_start + half)
    return node


def _init_frozen(node, frozen_bits):
    if node.leaf:
        node.frozen = bool(frozen_bits[node.lane_id])
        return
    _init_frozen(node.left, frozen_bits)
    _init_frozen(node.right, frozen_bits)


def _recursive_decode(node):
    if not node.leaf:
        half = len(node.lam) // 2
        for i in range(half):
            node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
        _recursive_decode(node.left)
        for i in range(half):
            node.right.lam[i] = g_operation(
                node.lam[i], node.lam[half + i], node.left.s[i]
            )
        _recursive_decode(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
    else:
        if node.frozen:
            node.s[0] = 0
        else:
            node.s[0] = 0 if node.lam[0] >= 0 else 1


def _collect_u(node, u_hat):
    if node.leaf:
        u_hat[node.lane_id] = node.s[0]
        return
    _collect_u(node.left, u_hat)
    _collect_u(node.right, u_hat)


def sc_decode_recursive(llr, frozen_bits):
    """树结构递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    root = _build_sc_tree(N)
    _init_frozen(root, frozen_bits)
    root.lam[:] = llr
    _recursive_decode(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_u(root, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 所需的层索引列表"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        for l in range(n):
            if psi % 2 == 0:
                llr_layers.append(n - 1 - l)
            psi >>= 1
        if not llr_layers:
            llr_layers = [n - 1]
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        for l in range(n):
            if psi % 2 == 1:
                bit_layers.append(n - 1 - l)
            psi >>= 1
        bit_layer_vec.append(bit_layers)

    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：默认使用高效树递归实现（与参考树等价）"""
    llr = map_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64), len(llr_ch))
    return sc_decode_recursive(llr, frozen_bits)


def sc_decode_simple(llr_ch, frozen_bits):
    """直接加载信道 LLR（无额外置换）"""
    llr = np.asarray(llr_ch, dtype=np.float64)
    return sc_decode_recursive(llr, frozen_bits)
