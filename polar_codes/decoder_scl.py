"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import sc_decode, llr_at_phi


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc8_byte(crc, byte, poly=0x07):
    crc ^= byte & 0xFF
    for _ in range(8):
        if crc & 0x80:
            crc = ((crc << 1) ^ poly) & 0xFF
        else:
            crc = (crc << 1) & 0xFF
    return crc


def _crc16_byte(crc, byte, poly=0x8005):
    crc ^= (byte & 0xFF) << 8
    for _ in range(8):
        if crc & 0x8000:
            crc = ((crc << 1) ^ poly) & 0xFFFF
        else:
            crc = (crc << 1) & 0xFFFF
    return crc


def _crc_process_bits(bits, crc_length):
    bits = [int(b) for b in bits]
    crc = 0
    bit_buffer = 0
    bit_count = 0
    byte_fn = _crc8_byte if crc_length == 8 else _crc16_byte

    for b in bits:
        bit_buffer = (bit_buffer << 1) | (b & 1)
        bit_count += 1
        if bit_count == 8:
            crc = byte_fn(crc, bit_buffer)
            bit_buffer = 0
            bit_count = 0

    if bit_count > 0:
        bit_buffer <<= (8 - bit_count)
        crc = byte_fn(crc, bit_buffer)
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    bits = [int(b) for b in info_bits]
    rem = _crc_process_bits(bits, crc_length)
    crc_bits = [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.array(bits + crc_bits, dtype=np.int8)


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return _crc_process_bits(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（基于递归 SC 的 LLR 计算）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _pm_add(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_internal = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [(np.zeros(self.N, dtype=np.int8), 0.0)]

        for phi in range(self.N):
            new_paths = []
            for u_hat, pm in paths:
                llr_phi = llr_at_phi(llr_internal, self.frozen_bits, u_hat, phi)
                if self.frozen_bits[phi]:
                    u_new = u_hat.copy()
                    u_new[phi] = 0
                    new_paths.append((u_new, self._pm_add(pm, llr_phi, 0)))
                else:
                    for bit in (0, 1):
                        u_new = u_hat.copy()
                        u_new[phi] = bit
                        new_paths.append((u_new, self._pm_add(pm, llr_phi, bit)))

            new_paths.sort(key=lambda x: x[1])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [
                (u, p) for u, p in paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best_u, best_pm = min(paths, key=lambda x: x[1])
        return best_u.copy(), best_pm
