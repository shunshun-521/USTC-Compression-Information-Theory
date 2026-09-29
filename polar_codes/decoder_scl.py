"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(expected, bits)


def _subtree_u_up(u_hat, offset, length):
    """由已判决比特重建子树顶层的部分和向量 u_up"""
    if length == 1:
        return u_hat[offset:offset + 1].copy()
    half = length // 2
    left_up = _subtree_u_up(u_hat, offset, half)
    right_up = _subtree_u_up(u_hat, offset + half, half)
    return np.concatenate([left_up ^ right_up, right_up])


def _llr_at_bit(llr_ch, frozen_bits, u_hat, phi):
    """计算第 phi 个比特的 LLR（已知 u_hat[0:phi]）"""

    def walk(llr_seg, frozen_seg, offset):
        n = len(llr_seg)
        if n == 1:
            return llr_seg[0]
        half = n // 2
        mid = offset + half
        if phi < mid:
            llr_left = f_operation(llr_seg[:half], llr_seg[half:])
            return walk(llr_left, frozen_seg[:half], offset)
        u_left_up = _subtree_u_up(u_hat, offset, half)
        llr_right = g_operation(llr_seg[:half], llr_seg[half:], u_left_up)
        return walk(llr_right, frozen_seg[half:], mid)

    return walk(llr_ch, frozen_bits, 0)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)

    @staticmethod
    def _pm_penalty(llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = _llr_at_bit(llr_ch, self.frozen_bits, path["u"], phi)
                if self.frozen_bits[phi]:
                    pm = path["pm"] + self._pm_penalty(llr, 0)
                    u = path["u"].copy()
                    u[phi] = 0
                    candidates.append({"pm": pm, "u": u})
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr, u_bit)
                        u = path["u"].copy()
                        u[phi] = u_bit
                        candidates.append({"pm": pm, "u": u})
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0 and self.info_indices is not None:
            seq_paths = []
            for p in paths:
                bits = p["u"][self.info_indices]
                if crc_check(bits, self.crc_length):
                    seq_paths.append(p)
            if seq_paths:
                paths = seq_paths

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
