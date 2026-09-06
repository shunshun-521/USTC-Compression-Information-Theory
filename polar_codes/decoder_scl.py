"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _update_bits,
    _update_llrs,
    active_bit_level,
    active_llr_level,
    bit_reversed,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _bits_to_bytes_msb(bits):
    bits = [int(b) for b in bits]
    nbytes = (len(bits) + 7) // 8
    padded = bits + [0] * (nbytes * 8 - len(bits))
    out = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | padded[i + j]
        out.append(byte)
    return bytes(out)


def _crc8_byte(data, poly=CRC8_POLY, init=0):
    crc = init
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_byte(data, poly=CRC16_POLY, init=0):
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    data = _bits_to_bytes_msb(info_bits)
    if crc_length == 8:
        remainder = _crc8_byte(data)
    else:
        remainder = _crc16_byte(data)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, path):
        new = _Path(self.N, self.n, path.L[:, 0])
        new.L = path.L.copy()
        new.B = path.B.copy()
        new.pm = path.pm
        new.u_hat = path.u_hat.copy()
        return new

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(l, path.L, path.B, self.n, self.N, use_min_sum=True)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path.pm += _pm_penalty(llr, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(l, new_path.B, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += _pm_penalty(llr, bit)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        _update_bits(l, new_path.B, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_valid = []
        for path in paths:
            if self.crc_length > 0:
                payload = path.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_valid.append(path)
            else:
                crc_valid.append(path)

        best = min(crc_valid or paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
