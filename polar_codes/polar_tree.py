"""aff3ct 风格极化码二叉树，用于 SC / SCL。"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation


class PolarNode:
    def __init__(self, size):
        self.size = size
        self.lane = None
        self.frozen = False
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.father = None
        self.left = None
        self.right = None
        if size > 1:
            half = size // 2
            self.left = PolarNode(half)
            self.right = PolarNode(half)
            self.left.father = self
            self.right.father = self

    def is_leaf(self):
        return self.size == 1

    def is_left(self):
        return self.father is not None and self is self.father.left

    def is_right(self):
        return self.father is not None and self is self.father.right

    def is_root(self):
        return self.father is None


def _init_frozen(node, frozen_bits, offset=0):
    if node.is_leaf():
        node.lane = offset
        node.frozen = bool(frozen_bits[offset])
        return offset + 1
    off = _init_frozen(node.left, frozen_bits, offset)
    return _init_frozen(node.right, frozen_bits, off)


def _collect_leaves(node, out):
    if node.is_leaf():
        out.append(node)
    else:
        _collect_leaves(node.left, out)
        _collect_leaves(node.right, out)


def compute_depth(leaf_index, m):
    return (leaf_index & (-leaf_index)).bit_length() - 1


def apply_f(node):
    half = node.size // 2
    for i in range(half):
        node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])


def apply_g(node):
    half = node.size // 2
    for i in range(half):
        node.right.lam[i] = g_operation(
            node.lam[i], node.lam[half + i], node.left.s[i]
        )


def compute_sums(node):
    half = node.size // 2
    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def recursive_compute_llr(leaf, depth):
    if depth != 0:
        recursive_compute_llr(leaf.father, depth - 1)
    if leaf.is_left():
        apply_f(leaf.father)
    if leaf.is_right():
        apply_g(leaf.father)


def propagate_sums(leaf):
    node = leaf
    if not node.is_leaf():
        compute_sums(node)
    if node.is_right() and not node.is_root():
        propagate_sums(node.father)


def recursive_decode(node):
    if node.is_leaf():
        node.s[0] = 0 if node.frozen else (0 if node.lam[0] >= 0 else 1)
        return
    apply_f(node)
    recursive_decode(node.left)
    apply_g(node)
    recursive_decode(node.right)
    compute_sums(node)


def _clone_node(node):
    new = PolarNode(node.size)
    new.lane = node.lane
    new.frozen = node.frozen
    new.lam = node.lam.copy()
    new.s = node.s.copy()
    if not node.is_leaf():
        new.left = _clone_node(node.left)
        new.right = _clone_node(node.right)
        new.left.father = new
        new.right.father = new
    return new


class PolarTree:
    def __init__(self, N, frozen_bits):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.root = PolarNode(N)
        _init_frozen(self.root, self.frozen_bits)
        self.leaves = []
        _collect_leaves(self.root, self.leaves)
        self.pm = 0.0

    def load(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        self.root.lam[:] = llr_ch[br]
        self.pm = 0.0

    def clone(self):
        other = PolarTree.__new__(PolarTree)
        other.N = self.N
        other.frozen_bits = self.frozen_bits
        other.root = _clone_node(self.root)
        other.leaves = []
        _collect_leaves(other.root, other.leaves)
        other.pm = self.pm
        return other

    def u_hat(self):
        u = np.zeros(self.N, dtype=int)
        for leaf in self.leaves:
            u[leaf.lane] = leaf.s[0]
        return u


def tree_sc_decode(llr_ch, frozen_bits):
    tree = PolarTree(len(llr_ch), frozen_bits)
    tree.load(llr_ch)
    recursive_decode(tree.root)
    return tree.u_hat()


def metric_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    return pm + (0.0 if bit == hard else abs(llr))
