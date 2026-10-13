"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from channel import map_llr_to_decoder
from decoder_sc import (
    _SCNode,
    _build_sc_tree,
    _init_frozen,
    f_operation,
    g_operation,
    sc_decode_recursive,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.uint8)
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8)
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in np.asarray(bits, dtype=np.uint8):
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


def _clone_node(node):
    new = _SCNode(len(node.lam), node.lane_id)
    new.lam = node.lam.copy()
    new.s = node.s.copy()
    new.frozen = node.frozen
    new.leaf = node.leaf
    if not node.leaf:
        new.left = _clone_node(node.left)
        new.right = _clone_node(node.right)
    return new


def _llr_at_lane(root, lane):
    def walk(node, target):
        if node.leaf:
            return float(node.lam[0])
        half = len(node.lam) // 2
        for i in range(half):
            node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
        if target < node.left.lane_id + half:
            return walk(node.left, target)
        for i in range(half):
            node.right.lam[i] = g_operation(
                node.lam[i], node.lam[half + i], node.left.s[i]
            )
        return walk(node.right, target)

    return walk(root, lane)


def _apply_decided_bit(root, lane, bit):
    node = root
    stack = []
    while not node.leaf:
        stack.append(node)
        half = len(node.lam) // 2
        if lane < node.left.lane_id + half:
            node = node.left
        else:
            node = node.right
    node.s[0] = bit
    for parent in reversed(stack):
        half = len(parent.lam) // 2
        for i in range(half):
            parent.s[i] = parent.left.s[i] ^ parent.right.s[i]
            parent.s[half + i] = parent.right.s[i]


class _Path:
    __slots__ = ("pm", "u", "root")

    def __init__(self, root, u, pm=0.0):
        self.pm = pm
        self.u = u
        self.root = root


class SCLDecoder:
    """SCL 译码器（增量树状态 + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = max(1, list_size)
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]
        template = _build_sc_tree(N)
        _init_frozen(template, self.frozen_bits)
        self._template = template

    def _pm_step(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr = map_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64), self.N)
        if self.L == 1 and self.crc_length == 0:
            return sc_decode_recursive(llr, self.frozen_bits), 0.0

        root0 = _clone_node(self._template)
        root0.lam[:] = llr
        paths = [_Path(root0, np.zeros(self.N, dtype=np.int8), 0.0)]

        for phi in range(self.N):
            new_paths = []
            for p in paths:
                llr_phi = _llr_at_lane(p.root, phi)
                if self.frozen_bits[phi]:
                    child_root = _clone_node(p.root)
                    _apply_decided_bit(child_root, phi, 0)
                    u2 = p.u.copy()
                    u2[phi] = 0
                    new_paths.append(
                        _Path(child_root, u2, self._pm_step(p.pm, llr_phi, 0))
                    )
                else:
                    for bit in (0, 1):
                        child_root = _clone_node(p.root)
                        _apply_decided_bit(child_root, phi, bit)
                        u2 = p.u.copy()
                        u2[phi] = bit
                        new_paths.append(
                            _Path(child_root, u2, self._pm_step(p.pm, llr_phi, bit))
                        )
            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.L]

        paths.sort(key=lambda x: x.pm)
        if self.crc_length > 0:
            for p in paths:
                info = p.u[self.info_idx]
                if crc_check(info, self.crc_length):
                    return p.u.astype(int), float(p.pm)
        return paths[0].u.astype(int), float(paths[0].pm)
