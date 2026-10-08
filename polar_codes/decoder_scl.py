"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _SCNode,
    _apply_f_on_parent,
    _apply_g_on_parent,
    _build_tree,
    _get_leaves_ordered,
    _init_frozen,
    _propagate_sums_up,
    sc_decode_recursive,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


def _path_metric(pm, llr, bit):
    consistent = (bit == 0 and llr >= 0) or (bit == 1 and llr < 0)
    return pm + (0.0 if consistent else abs(llr))


def _leaf_llr(llr_ch, frozen_bits, u_partial, lane_id, template_root):
    root = _clone_tree(template_root)
    root.lam = np.asarray(llr_ch, dtype=np.float64).copy()
    leaves = _get_leaves_ordered(root)
    for leaf in leaves:
        if leaf.lane_id < lane_id:
            leaf.s = np.array([u_partial[leaf.lane_id]], dtype=np.int8)
            _propagate_sums_up(leaf)
    return _compute_llr_to_leaf(root, lane_id)


def _clone_tree(node):
    new = _SCNode(node.size, node.lane_id)
    new.is_frozen = node.is_frozen
    if node.size > 1:
        new.left = _clone_tree(node.left)
        new.right = _clone_tree(node.right)
        new.left.parent = new
        new.right.parent = new
    return new


def _contains_lane(node, lane):
    if node.size == 1:
        return node.lane_id == lane
    return _contains_lane(node.left, lane) or _contains_lane(node.right, lane)


def _compute_llr_to_leaf(root, target_lane):
    def recurse(node):
        if node.size == 1:
            return float(node.lam[0])
        _apply_f_on_parent(node)
        if _contains_lane(node.left, target_lane):
            return recurse(node.left)
        _apply_g_on_parent(node)
        return recurse(node.right)

    return recurse(root)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int) if info_indices is not None else None
        )
        lane_counter = [0]
        self._template = _build_tree(N, lane_counter)
        _init_frozen(self._template, self.frozen_bits)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        active = [(0.0, np.zeros(self.N, dtype=np.int32))]
        for lane in range(self.N):
            expanded = []
            for pm, u in active:
                llr_bit = _leaf_llr(
                    llr_ch, self.frozen_bits, u, lane, self._template
                )
                if self.frozen_bits[lane]:
                    u_new = u.copy()
                    u_new[lane] = 0
                    expanded.append((_path_metric(pm, llr_bit, 0), u_new))
                else:
                    for bit in (0, 1):
                        u_new = u.copy()
                        u_new[lane] = bit
                        expanded.append((_path_metric(pm, llr_bit, bit), u_new))
            expanded.sort(key=lambda x: x[0])
            active = expanded[: self.list_size]

        best_pm, best_u = active[0]
        if self.crc_length > 0:
            valid = []
            for pm, u in active:
                bits = (
                    u[self.info_indices] if self.info_indices is not None else u
                )
                if crc_check(bits, self.crc_length):
                    valid.append((pm, u))
            if valid:
                best_pm, best_u = min(valid, key=lambda x: x[0])

        return best_u, best_pm
