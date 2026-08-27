"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _SCNode,
    _build_tree,
    _collect_decoded_bits,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_run(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_run(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_run(bits, poly, crc_length) == 0


def _clone_tree(node):
    new = _SCNode(node.size, node.lane_id, node.is_frozen)
    new.lambda_ = node.lambda_.copy()
    new.s = node.s.copy()
    if node.left is not None:
        new.left = _clone_tree(node.left)
        new.right = _clone_tree(node.right)
    return new


def _recursive_decode_path(node):
    if node.left is not None:
        half = node.size // 2
        for i in range(half):
            node.left.lambda_[i] = f_operation(
                node.lambda_[i], node.lambda_[half + i]
            )
        _recursive_decode_path(node.left)
        for i in range(half):
            node.right.lambda_[i] = g_operation(
                node.lambda_[i],
                node.lambda_[half + i],
                node.left.s[i],
            )
        _recursive_decode_path(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
    else:
        if not node.is_frozen:
            node.s[0] = 1 if node.lambda_[0] < 0 else 0


def _lane_order(node):
    if node.left is None:
        return [node.lane_id]
    return _lane_order(node.left) + _lane_order(node.right)


def _find_leaf(node, lane):
    if node.left is None:
        return node
    left_lanes = _lane_order(node.left)
    if lane in left_lanes:
        return _find_leaf(node.left, lane)
    return _find_leaf(node.right, lane)


def _apply_fixed_bits(root, fixed_bits):
    for lane, bit in fixed_bits.items():
        _find_leaf(root, lane).s[0] = bit


class SCLDecoder:
    """SCL 译码器（逐比特路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        root, _ = _build_tree(self.n, self.frozen_bits)
        self._lane_order = _lane_order(root)

    def _compute_leaf_llr(self, llr_ch, fixed_bits, lane):
        root, _ = _build_tree(self.n, self.frozen_bits)
        root.lambda_ = np.asarray(llr_ch, dtype=np.float64)
        _apply_fixed_bits(root, fixed_bits)
        _recursive_decode_path(root)
        return _find_leaf(root, lane).lambda_[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size
        paths = [{"bits": {}, "pm": 0.0}]

        for lane in self._lane_order:
            candidates = []
            for path in paths:
                llr = self._compute_leaf_llr(llr_ch, path["bits"], lane)
                if self.frozen_bits[lane]:
                    bit = 0
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    bits = path["bits"].copy()
                    bits[lane] = bit
                    candidates.append({"bits": bits, "pm": path["pm"] + penalty})
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
                        bits = path["bits"].copy()
                        bits[lane] = bit
                        candidates.append({"bits": bits, "pm": path["pm"] + penalty})

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:L]

        best = paths[0]
        u_hat = np.zeros(self.N, dtype=int)
        for lane, bit in best["bits"].items():
            u_hat[lane] = bit

        if self.crc_length > 0:
            valid = []
            for p in paths:
                uh = np.zeros(self.N, dtype=int)
                for lane, bit in p["bits"].items():
                    uh[lane] = bit
                if crc_check(uh[self.info_indices], self.crc_length):
                    valid.append((p["pm"], uh))
            if valid:
                valid.sort(key=lambda x: x[0])
                u_hat = valid[0][1]
                best = {"pm": valid[0][0]}

        return u_hat, best["pm"]
