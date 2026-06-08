"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    _build_tree,
    _collect_bits,
    _init_frozen,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError('crc_length must be 8 or 16')

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length))


def _apply_left(node):
    half = len(node.lambda_) // 2
    for i in range(half):
        node.left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[half + i])


def _apply_right(node):
    half = len(node.lambda_) // 2
    for i in range(half):
        node.right.lambda_[i] = g_operation(
            node.lambda_[i], node.lambda_[half + i], node.left.s[i]
        )


def _combine(node):
    half = len(node.lambda_) // 2
    for i in range(half):
        node.s[i] = (node.left.s[i] ^ node.right.s[i]) % 2
        node.s[half + i] = node.right.s[i]


def _node_path(template_root, target, path=None):
    if path is None:
        path = []
    if template_root is target:
        return path
    if _is_descendant(template_root.left, target):
        return _node_path(template_root.left, target, path + [0])
    return _node_path(template_root.right, target, path + [1])


def _is_descendant(node, target):
    if node is None:
        return False
    if node is target:
        return True
    return _is_descendant(node.left, target) or _is_descendant(node.right, target)


def _follow_path(root, path):
    node = root
    for d in path:
        node = node.left if d == 0 else node.right
    return node


def _scl_tree(node, states, list_size, template_root):
    """states: list of (tree_root, pm)，node 为模板树中的当前子树根。"""
    path = _node_path(template_root, node)
    if node.leaf:
        out = []
        for root, pm in states:
            cur = _follow_path(root, path)
            llr = cur.lambda_[0]
            if cur.frozen:
                pen = 0.0 if llr >= 0 else abs(llr)
                cur.s[0] = 0
                out.append((root, pm + pen))
            else:
                hard = 0 if llr >= 0 else 1
                for bit in (0, 1):
                    pen = 0.0 if bit == hard else abs(llr)
                    if bit == hard:
                        cur.s[0] = bit
                        out.append((root, pm + pen))
                    else:
                        nroot = copy.deepcopy(root)
                        _follow_path(nroot, path).s[0] = bit
                        out.append((nroot, pm + pen))
        out.sort(key=lambda x: x[1])
        return out[:list_size]

    new_states = []
    for root, pm in states:
        _apply_left(_follow_path(root, path))
        new_states.append((root, pm))

    left_states = _scl_tree(node.left, new_states, list_size, template_root)

    mid = []
    for root, pm in left_states:
        _apply_right(_follow_path(root, path))
        mid.append((root, pm))

    right_states = _scl_tree(node.right, mid, list_size, template_root)

    final = []
    for root, pm in right_states:
        _combine(_follow_path(root, path))
        final.append((root, pm))
    return final


class SCLDecoder:
    """SCL 译码器（树形列表译码）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self._template = _build_tree(self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        root = copy.deepcopy(self._template)
        _init_frozen(root, self.frozen_bits)
        root.lambda_[:] = llr_ch

        states = _scl_tree(self._template, [(root, 0.0)], self.list_size, self._template)

        candidates = []
        for r, pm in states:
            u_hat = np.zeros(self.N, dtype=int)
            _collect_bits(r, u_hat)
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            valid = [c for c in candidates if crc_check(c[1], self.crc_length)]
            pool = valid if valid else candidates
        else:
            pool = candidates

        best = min(pool, key=lambda x: x[0])
        return best[1].copy(), best[0]
