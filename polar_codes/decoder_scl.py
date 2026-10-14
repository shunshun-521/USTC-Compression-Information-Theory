"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _TreeNode,
    _build_tree,
    _collect_leaves,
    f_operation,
    g_operation,
    sc_decode_recursive,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _clone_tree(node):
    if node is None:
        return None
    new = _TreeNode(len(node.lam), frozen=node.frozen, leaf=node.leaf, lane=node.lane)
    new.lam = node.lam.copy()
    new.s = node.s.copy()
    new.left = _clone_tree(node.left)
    new.right = _clone_tree(node.right)
    return new


def _gather_leaves(node):
    if node.leaf:
        return [node]
    return _gather_leaves(node.left) + _gather_leaves(node.right)


def _propagate_sums(node):
    if node.leaf:
        return
    half = len(node.lam) // 2
    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def _compute_llr_to_leaf(root, leaf):
    """自根到目标叶子计算 LLR（沿父链执行 f/g）。"""
    path = []
    cur = leaf
    while cur is not root:
        parent = None
        # find parent by walk from root (tree is small)
        def find_parent(n, target, p=None):
            if n is target:
                return p
            if n.leaf:
                return None
            r = find_parent(n.left, target, n)
            if r is not None:
                return r
            return find_parent(n.right, target, n)

        parent = find_parent(root, cur)
        path.append((parent, cur))
        cur = parent
    path.reverse()

    for parent, child in path:
        half = len(parent.lam) // 2
        if child is parent.left:
            for i in range(half):
                parent.left.lam[i] = f_operation(parent.lam[i], parent.lam[half + i])
        else:
            for i in range(half):
                parent.right.lam[i] = g_operation(
                    parent.lam[i], parent.lam[half + i], parent.left.s[i]
                )
    return leaf.lam[0]


def _pm_add(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if int(u) != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """基于极化树的 SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        n = self.m
        root_template = _build_tree(n, self.frozen_bits)
        leaves_template = _gather_leaves(root_template)

        paths = []
        root0 = _clone_tree(root_template)
        root0.lam[:] = llr_ch
        paths.append({"pm": 0.0, "root": root0})

        for leaf in leaves_template:
            candidates = []
            for path in paths:
                llr_bit = _compute_llr_to_leaf(path["root"], _find_leaf(path["root"], leaf.lane))
                if leaf.frozen:
                    pm = _pm_add(path["pm"], llr_bit, 0)
                    new_root = _clone_tree(path["root"])
                    lf = _find_leaf(new_root, leaf.lane)
                    lf.s[0] = 0
                    _propagate_up(new_root, lf)
                    candidates.append((pm, new_root))
                else:
                    for u in (0, 1):
                        pm = _pm_add(path["pm"], llr_bit, u)
                        new_root = _clone_tree(path["root"])
                        lf = _find_leaf(new_root, leaf.lane)
                        lf.s[0] = u
                        _propagate_up(new_root, lf)
                        candidates.append((pm, new_root))

            candidates.sort(key=lambda x: x[0])
            paths = [{"pm": pm, "root": r} for pm, r in candidates[: self.list_size]]

        best = min(paths, key=lambda p: p["pm"])
        u_hat = np.zeros(self.N, dtype=np.int32)
        _collect_leaves(best["root"], u_hat)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                uh = np.zeros(self.N, dtype=np.int32)
                _collect_leaves(p["root"], uh)
                if crc_check(uh[self.info_indices], self.crc_length):
                    valid.append((p["pm"], uh))
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat, pm

        return u_hat, best["pm"]


def _find_leaf(root, lane):
    if root.leaf:
        return root if root.lane == lane else None
    left = _find_leaf(root.left, lane)
    if left is not None:
        return left
    return _find_leaf(root.right, lane)


def _propagate_up(root, leaf):
    """从叶子向上合并 s（与 aff3ct propagate_sums 一致）。"""
    stack = []
    cur = leaf
    while cur is not root:
        parent = _find_parent(root, cur)
        stack.append(parent)
        cur = parent
    for parent in stack:
        _propagate_sums(parent)


def _find_parent(root, child):
    if root.leaf:
        return None
    if root.left is child or root.right is child:
        return root
    p = _find_parent(root.left, child)
    if p is not None:
        return p
    return _find_parent(root.right, child)
