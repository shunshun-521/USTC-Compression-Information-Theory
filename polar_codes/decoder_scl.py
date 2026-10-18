"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
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
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _llr_at_bit(llr, frozen_bits, known_bits, phi):
    """在已知前缀 known_bits[0:phi] 时，计算第 phi 位的 LLR。"""

    def decode_node(llr_node, frozen_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            if idx < phi:
                u = known_bits[idx]
            elif idx == phi:
                return None, llr_node[0]
            else:
                u = 0
            return np.array([u], dtype=np.int8), None

        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        fr1 = frozen_node[:half]
        fr2 = frozen_node[half:]

        llr_up = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _up_decode(llr_up, fr1, offset)
        if phi < offset + half:
            return _llr_at_bit_inner(llr_up, fr1, known_bits, phi, offset)

        llr_low = g_operation(llr1, llr2, u_hat1_up)
        return _llr_at_bit_inner(llr_low, fr2, known_bits, phi, offset + half)

    def _up_decode(llr_node, frozen_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            if idx < phi:
                u = known_bits[idx]
            else:
                u = 0
            return np.array([u], dtype=np.int8), np.array([u], dtype=np.int8)
        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        fr1 = frozen_node[:half]
        fr2 = frozen_node[half:]
        llr_up = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _up_decode(llr_up, fr1, offset)
        llr_low = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _up_decode(llr_low, fr2, offset + half)
        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = np.bitwise_xor(u_hat1_up.astype(np.int8), u_hat2_up.astype(np.int8))
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    def _llr_at_bit_inner(llr_node, frozen_node, known_bits, phi, offset):
        n = len(llr_node)
        if n == 1:
            return llr_node[0]
        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        fr1 = frozen_node[:half]
        fr2 = frozen_node[half:]
        llr_up = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _up_decode(llr_up, fr1, offset)
        if phi < offset + half:
            return _llr_at_bit_inner(llr_up, fr1, known_bits, phi, offset)
        llr_low = g_operation(llr1, llr2, u_hat1_up)
        return _llr_at_bit_inner(llr_low, fr2, known_bits, phi, offset + half)

    return _llr_at_bit_inner(llr, frozen_bits, known_bits, phi, 0)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool).ravel()
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64).ravel()
        paths = [(0.0, np.zeros(self.N, dtype=np.int8))]

        for phi in range(self.N):
            candidates = []
            for pm, bits in paths:
                llr_phi = _llr_at_bit(llr_ch, self.frozen_bits, bits, phi)
                if self.frozen_bits[phi]:
                    b2 = bits.copy()
                    b2[phi] = 0
                    candidates.append((pm + self._pm_penalty(llr_phi, 0), b2))
                else:
                    for u in (0, 1):
                        b2 = bits.copy()
                        b2[phi] = u
                        candidates.append((pm + self._pm_penalty(llr_phi, u), b2))
            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda x: x[0])
        if self.crc_length > 0:
            valid = []
            for pm, bits in paths:
                if crc_check(bits[self.info_indices], self.crc_length):
                    valid.append((pm, bits))
            if valid:
                best = min(valid, key=lambda x: x[0])

        return best[1].copy(), best[0]
