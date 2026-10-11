"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _hard_bit,
    _build_tree,
    _allocate_tree,
    _init_frozen,
    _collect_u,
    SCDecoder,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_digest(bits, crc_length):
    """Reflected CRC（多项式 0x07 / 0x8005），用于 CA-SCL。"""
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    h = 0
    mask = (1 << crc_length) - 1
    for bit in np.asarray(bits, dtype=int).ravel():
        h ^= int(bit)
        for _ in range(8):
            if h & 1:
                h = (h >> 1) ^ poly
            else:
                h >>= 1
    return h & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    rem = _crc_digest(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    return _crc_digest(bits, crc_length) == 0


def _compute_depth(index, tree_depth):
    if index == 0:
        return tree_depth - 1
    res = 0
    while (index & 1) != 1 and res <= tree_depth - 1:
        index >>= 1
        res += 1
    return res


def _link_parents(node, parent=None):
    node._parent = parent
    if not node.is_leaf:
        _link_parents(node.left, node)
        _link_parents(node.right, node)


def _get_leaves(root):
    leaves = []

    def walk(n):
        if n.is_leaf:
            leaves.append(n)
        else:
            walk(n.left)
            walk(n.right)

    walk(root)
    leaves.sort(key=lambda x: x.lane_id)
    return leaves


def _apply_f(parent):
    half = parent.size // 2
    parent.left.lambda_[:] = f_operation(parent.lambda_[:half], parent.lambda_[half:])


def _apply_g(parent):
    half = parent.size // 2
    parent.right.lambda_[:] = g_operation(
        parent.lambda_[:half], parent.lambda_[half:], parent.left.s
    )


def _compute_sums(parent):
    half = parent.size // 2
    parent.s[:half] = (parent.left.s ^ parent.right.s) % 2
    parent.s[half:] = parent.right.s


def _recursive_compute_llr(leaf, depth):
    if depth != 0:
        _recursive_compute_llr(leaf._parent, depth - 1)
    parent = leaf._parent
    if parent is None:
        return
    if leaf is parent.left:
        _apply_f(parent)
    if leaf is parent.right:
        _apply_g(parent)


def _propagate_sums(leaf):
    node = leaf
    while node is not None and node._parent is not None:
        if node is node._parent.right:
            _compute_sums(node._parent)
            node = node._parent
        else:
            break


def _path_metric(pm, llr, u):
    if u != _hard_bit(llr):
        pm += abs(llr)
    return pm


def _clone_tree(root):
    return copy.deepcopy(root)


class SCLDecoder:
    """SCL 译码器（L 条路径，树拷贝实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self._br = bit_reversal_permutation(N)

    def _new_root(self, llr_dec):
        root = _build_tree(self.N, [0])
        _allocate_tree(root)
        _init_frozen(root, self.frozen_bits)
        _link_parents(root)
        root.lambda_[:] = llr_dec
        return root

    def decode(self, llr_ch):
        if self.list_size == 1:
            u = SCDecoder(self.N, self.frozen_bits).decode(llr_ch)
            return u, 0.0

        llr_dec = np.asarray(llr_ch, dtype=np.float64)[self._br]
        paths = [{"root": self._new_root(llr_dec), "pm": 0.0}]
        paths[0]["leaves"] = _get_leaves(paths[0]["root"])

        for phi in range(self.N):
            expanded = []
            for path in paths:
                leaf = path["leaves"][phi]
                _recursive_compute_llr(leaf, _compute_depth(phi, self.m))
                llr_phi = leaf.lambda_[0]
                if self.frozen_bits[phi]:
                    leaf.s[0] = 0
                    path["pm"] = _path_metric(path["pm"], llr_phi, 0)
                    expanded.append(path)
                else:
                    for u in (0, 1):
                        new_root = _clone_tree(path["root"])
                        new_leaves = _get_leaves(new_root)
                        new_leaves[phi].s[0] = u
                        pm = _path_metric(path["pm"], llr_phi, u)
                        expanded.append({"root": new_root, "leaves": new_leaves, "pm": pm})

            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]
            for path in paths:
                _propagate_sums(path["leaves"][phi])

        best = min(paths, key=lambda p: p["pm"])
        u_hat = np.zeros(self.N, dtype=int)
        _collect_u(best["root"], u_hat)

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for path in paths:
                uh = np.zeros(self.N, dtype=int)
                _collect_u(path["root"], uh)
                if crc_check(uh[info_idx], self.crc_length):
                    valid.append((path["pm"], uh))
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat, pm

        return u_hat, best["pm"]
