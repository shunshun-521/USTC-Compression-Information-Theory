"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _TreeNode,
    _build_tree,
    _alloc_tree,
    _set_frozen,
    f_operation,
    g_operation,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _leaf_for_lane(node, lane):
    if node.left is None:
        return node if node.lane == lane else None
    left = _leaf_for_lane(node.left, lane)
    return left if left is not None else _leaf_for_lane(node.right, lane)


def _run_sc_prefix(root, lane_limit, decisions, frozen_bits):
    """顺序 SC 至 lane_limit（含），decisions 写入已判决比特。"""

    def visit(node):
        if node.left is None:
            lane = node.lane
            if lane > lane_limit:
                return
            if frozen_bits[lane]:
                node.s[0] = 0
            else:
                node.s[0] = int(decisions[lane])
            return

        half = len(node.lam) // 2
        for i in range(half):
            node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
        visit(node.left)
        for i in range(half):
            node.right.lam[i] = g_operation(
                node.lam[i], node.lam[half + i], node.left.s[i]
            )
        visit(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]

    visit(root)


def _lane_llr(llr_perm, frozen_bits, decisions, lane):
    """已知 decisions[0:lane] 时，计算 lane 处叶子 LLR。"""
    N = len(llr_perm)
    n = int(math.log2(N))
    root = _build_tree(n)
    _alloc_tree(root, N)
    _set_frozen(root, frozen_bits)
    for i in range(N):
        root.lam[i] = float(llr_perm[i])
    _run_sc_prefix(root, lane - 1, decisions, frozen_bits)
    leaf = _leaf_for_lane(root, lane)
    if leaf is None:
        return 0.0
    # 仅计算 lane 的 f/g 链
    node = root
    path = []
    while node.left is not None:
        half = len(node.lam) // 2
        if lane <= _max_lane(node.left):
            for i in range(half):
                node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
            _run_sc_prefix(node.left, lane - 1, decisions, frozen_bits)
            path.append(("L", node))
            node = node.left
        else:
            for i in range(half):
                node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
            _run_sc_prefix(node.left, _max_lane(node.left), decisions, frozen_bits)
            for i in range(half):
                node.right.lam[i] = g_operation(
                    node.lam[i], node.lam[half + i], node.left.s[i]
                )
            path.append(("R", node))
            node = node.right
    return float(node.lam[0])


def _max_lane(node):
    if node.left is None:
        return node.lane
    return max(_max_lane(node.left), _max_lane(node.right))


def _pm_add(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    return pm if hard == bit else pm + abs(llr)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.br = bit_reversal_permutation(N)
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [(0.0, np.zeros(self.N, dtype=np.int8))]

        for lane in range(self.N):
            candidates = []
            for pm, bits in paths:
                llr_leaf = _lane_llr(llr_perm, self.frozen_bits, bits, lane)
                if self.frozen_bits[lane]:
                    bit = 0
                    bits_new = bits.copy()
                    bits_new[lane] = bit
                    candidates.append((_pm_add(pm, llr_leaf, bit), bits_new))
                else:
                    for bit in (0, 1):
                        bits_new = bits.copy()
                        bits_new[lane] = bit
                        candidates.append((_pm_add(pm, llr_leaf, bit), bits_new))
            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                (pm, bits)
                for pm, bits in paths
                if crc_check(bits[self.info_idx], self.crc_length)
            ]
            if valid:
                paths = valid

        pm, bits = min(paths, key=lambda x: x[0])
        return bits.copy(), float(pm)
