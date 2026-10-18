"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    _build_tree,
    _init_frozen,
    _recursive_decode,
    sc_decode,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int)

    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) & 0xFF) ^ _CRC8_POLY
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        reg = 0xFFFF
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) & 0xFFFF) ^ _CRC16_POLY
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    return np.array_equal(bits[-crc_length:], crc_encode(payload, crc_length)[-crc_length:])


def _deep_copy_tree(node):
    new = copy.copy(node)
    new.lambda_ = node.lambda_.copy()
    new.s = node.s.copy()
    if node.size > 1:
        new.left = _deep_copy_tree(node.left)
        new.right = _deep_copy_tree(node.right)
    return new


def _leaf_llr_for_prefix(root, phi):
    """在已译码 0..phi-1 后，返回第 phi 位的 LLR（叶节点 lambda）"""

    def walk(node, bit_lo, bit_hi):
        if bit_hi - bit_lo == 1:
            return node.lambda_[0]
        mid = (bit_lo + bit_hi) // 2
        if phi < mid:
            return walk(node.left, bit_lo, mid)
        return walk(node.right, mid, bit_hi)

    return walk(root, 0, root.size)


def _decode_prefix(root, u_prefix):
    """将 u[0:len(u_prefix)) 固定到树中并更新内部状态"""
    u_hat = np.zeros(root.size, dtype=int)
    u_hat[: len(u_prefix)] = u_prefix
    root.lambda_  # 保持信道 LLR
    # 重新从根译码，但在叶层 0..len-1 强制为 u_prefix
    _recursive_decode_forced(root, u_hat, end_phi=len(u_prefix))


def _recursive_decode_forced(node, u_hat, end_phi):
    """译码至 end_phi（不含），叶 < end_phi 使用 u_hat 强制值"""
    if node.size > 1:
        half = node.size // 2
        left, right = node.left, node.right
        from decoder_sc import f_operation, g_operation

        for i in range(half):
            left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[half + i])
        _recursive_decode_forced(left, u_hat, end_phi)
        for i in range(half):
            right.lambda_[i] = g_operation(
                node.lambda_[i], node.lambda_[half + i], left.s[i]
            )
        _recursive_decode_forced(right, u_hat, end_phi)
        for i in range(half):
            node.s[i] = left.s[i] ^ right.s[i]
            node.s[half + i] = right.s[i]
    else:
        idx = node.lane_id
        if idx < end_phi:
            node.s[0] = u_hat[idx]
        elif node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = 0 if node.lambda_[0] >= 0 else 1


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        lane_counter = [0]
        self._template_root = _build_tree(N, lane_counter)
        _init_frozen(self._template_root, self.frozen_bits)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _bit_llr(self, llr_ch, u_prefix, phi):
        root = _deep_copy_tree(self._template_root)
        root.lambda_[:] = llr_ch
        if phi > 0:
            _decode_prefix(root, u_prefix[:phi])
        return _leaf_llr_for_prefix(root, phi)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [(np.zeros(self.N, dtype=int), 0.0)]
        for phi in range(self.N):
            new_paths = []
            for u_hat, pm in paths:
                llr_bit = self._bit_llr(llr_ch, u_hat, phi)
                if self.frozen_bits[phi]:
                    bit = 0
                    pen = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    u2 = u_hat.copy()
                    u2[phi] = bit
                    new_paths.append((u2, pm + pen))
                else:
                    for bit in (0, 1):
                        expected = 0 if llr_bit >= 0 else 1
                        pen = 0.0 if bit == expected else abs(llr_bit)
                        u2 = u_hat.copy()
                        u2[phi] = bit
                        new_paths.append((u2, pm + pen))
            new_paths.sort(key=lambda x: x[1])
            paths = new_paths[: self.list_size]

        best_u, best_pm = paths[0]
        if self.crc_length > 0:
            info_bits = best_u[self.info_indices]
            valid = [
                p
                for p in paths
                if crc_check(p[0][self.info_indices], self.crc_length)
            ]
            if valid:
                best_u, best_pm = min(valid, key=lambda x: x[1])
        return best_u, best_pm
