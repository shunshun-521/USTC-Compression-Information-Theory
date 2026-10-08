"""
极化码 SC（串行抵消）译码器
树形递归实现（参考 aff3ct Decoder_polar_SC_naive）
"""
import math

import numpy as np


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _hard_bit(llr):
    return 0 if llr >= 0 else 1


class _SCNode:
    __slots__ = ("size", "lane_id", "left", "right", "parent", "lam", "s", "is_frozen")

    def __init__(self, size, lane_id=None):
        self.size = size
        self.lane_id = lane_id
        self.left = None
        self.right = None
        self.parent = None
        self.lam = None
        self.s = None
        self.is_frozen = False


def _build_tree(size, lane_counter):
    node = _SCNode(size)
    if size == 1:
        node.lane_id = lane_counter[0]
        lane_counter[0] += 1
        return node
    half = size // 2
    node.left = _build_tree(half, lane_counter)
    node.right = _build_tree(half, lane_counter)
    node.left.parent = node
    node.right.parent = node
    return node


def _init_frozen(node, frozen_bits):
    if node.size == 1:
        node.is_frozen = bool(frozen_bits[node.lane_id])
        return
    _init_frozen(node.left, frozen_bits)
    _init_frozen(node.right, frozen_bits)


def _get_leaves_ordered(node, leaves=None):
    if leaves is None:
        leaves = []
    if node.size == 1:
        leaves.append(node)
        return leaves
    _get_leaves_ordered(node.left, leaves)
    _get_leaves_ordered(node.right, leaves)
    return leaves


def _apply_f_on_parent(parent):
    half = parent.size // 2
    parent.left.lam = np.array(
        [f_operation(parent.lam[i], parent.lam[half + i]) for i in range(half)],
        dtype=np.float64,
    )


def _apply_g_on_parent(parent):
    half = parent.size // 2
    parent.right.lam = np.array(
        [
            g_operation(parent.lam[i], parent.lam[half + i], parent.left.s[i])
            for i in range(half)
        ],
        dtype=np.float64,
    )


def _propagate_sums_up(leaf):
    node = leaf
    while node.parent is not None:
        parent = node.parent
        half = parent.size // 2
        if parent.left.s is not None and parent.right.s is not None:
            parent.s = np.zeros(parent.size, dtype=np.int8)
            parent.s[:half] = parent.left.s ^ parent.right.s
            parent.s[half:] = parent.right.s
        node = parent


def _recursive_decode(node):
    if node.size == 1:
        node.s = np.zeros(1, dtype=np.int8)
        if not node.is_frozen:
            node.s[0] = _hard_bit(node.lam[0])
        return

    half = node.size // 2
    _apply_f_on_parent(node)
    _recursive_decode(node.left)

    _apply_g_on_parent(node)
    _recursive_decode(node.right)

    node.s = np.zeros(node.size, dtype=np.int8)
    node.s[:half] = node.left.s ^ node.right.s
    node.s[half:] = node.right.s


def sc_decode_recursive(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    lane_counter = [0]
    root = _build_tree(N, lane_counter)
    _init_frozen(root, frozen_bits)
    root.lam = llr.copy()
    _recursive_decode(root)
    u_hat = np.zeros(N, dtype=np.int32)
    for leaf in _get_leaves_ordered(root):
        u_hat[leaf.lane_id] = leaf.s[0]
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n - 1, -1, -1))
        else:
            llr_layers = []
            psi = phi
            while psi & 1:
                llr_layers.append(n - 1 - len(llr_layers))
                psi >>= 1
        bit_layers = []
        psi = phi
        while psi & 1:
            bit_layers.append(n - 1 - len(bit_layers))
            psi >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    lambda_offset = np.zeros(n + 2, dtype=int)
    for i in range(1, n + 2):
        lambda_offset[i] = 2 ** (n - i + 1)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_recursive(llr_ch, frozen_bits)
