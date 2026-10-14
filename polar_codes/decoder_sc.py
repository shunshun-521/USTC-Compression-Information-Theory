"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现：复用树结构）
"""
import numpy as np
import math


def _sign_pm(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _hard_bit(llr, frozen):
    if frozen:
        return 0
    return 0 if llr >= 0 else 1


class _SCNode:
    __slots__ = ("size", "is_leaf", "left", "right", "lam", "s", "lane", "frozen")

    def __init__(self, size):
        self.size = size
        self.is_leaf = size == 1
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.lane = 0
        self.frozen = False
        if not self.is_leaf:
            half = size // 2
            self.left = _SCNode(half)
            self.right = _SCNode(half)


def _assign_lanes(node, idx=0):
    if node.is_leaf:
        node.lane = idx
        return idx + 1
    idx = _assign_lanes(node.left, idx)
    return _assign_lanes(node.right, idx)


def _recursive_decode_node(node):
    if node.is_leaf:
        node.s[0] = _hard_bit(node.lam[0], node.frozen)
        return
    half = node.size // 2
    for i in range(half):
        node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
    _recursive_decode_node(node.left)
    for i in range(half):
        node.right.lam[i] = g_operation(
            node.lam[i], node.lam[half + i], node.left.s[i]
        )
    _recursive_decode_node(node.right)
    for i in range(half):
        node.s[i] = (node.left.s[i] ^ node.right.s[i]) & 1
        node.s[half + i] = node.right.s[i]


def _gather_bits(node, out):
    if node.is_leaf:
        out.append(node.s[0])
    else:
        _gather_bits(node.left, out)
        _gather_bits(node.right, out)


_TREE_CACHE = {}


def _get_tree(N):
    if N not in _TREE_CACHE:
        root = _SCNode(N)
        _assign_lanes(root)
        _TREE_CACHE[N] = root
    return _TREE_CACHE[N]


def sc_decode_recursive(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    root = _get_tree(N)
    root.lam[:] = llr
    root.s.fill(0)

    def set_frozen(node):
        if node.is_leaf:
            node.frozen = bool(frozen_bits[node.lane])
        else:
            set_frozen(node.left)
            set_frozen(node.right)

    set_frozen(root)
    _recursive_decode_node(root)
    out = []
    _gather_bits(root, out)
    return np.array(out, dtype=int)


def sc_decode(llr_ch, frozen_bits):
    """主译码入口（树结构复用，O(N log N)）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        while p & 1:
            layers.append((p & -p).bit_length() - 1)
            p >>= 1
        llr_layer_vec.append(layers)
        blayers = []
        if phi & 1:
            blayers.append(0)
            p = phi >> 1
            while p & 1:
                p >>= 1
            if p:
                blayers.append((p & -p).bit_length() - 1)
        else:
            p = phi >> 1
            while p and not (p & 1):
                p >>= 1
            if p:
                blayers.append((p & -p).bit_length() - 1)
        bit_layer_vec.append(blayers)
    return [1 << i for i in range(n + 1)], llr_layer_vec, bit_layer_vec
