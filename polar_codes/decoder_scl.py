"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode
from encoder import polar_encode

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_len - 1)
        for _ in range(crc_len):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
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


class _Path:
    __slots__ = ("pm", "u_hat")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器。"""

    _G_CACHE = {}

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self._info_idx = np.where(~self.frozen_bits)[0]
        if N not in SCLDecoder._G_CACHE:
            from encoder import build_generator_matrix
            SCLDecoder._G_CACHE[N] = build_generator_matrix(N)
        self._G = SCLDecoder._G_CACHE[N]

    def _bit_llr(self, u_partial, phi, llr_ch):
        x = polar_encode(u_partial)
        delta = self._G[phi, :].astype(np.float64)
        corr0 = np.dot(llr_ch, 1.0 - 2.0 * x)
        corr1 = np.dot(llr_ch, 1.0 - 2.0 * ((x + delta) % 2))
        return float(corr0 - corr1)

    def _penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [_Path(self.N)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr = self._bit_llr(path.u_hat, phi, llr_ch)
                bits = [0] if self.frozen_bits[phi] else [0, 1]
                for bit in bits:
                    child = _Path(self.N)
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[phi] = bit
                    child.pm = path.pm + self._penalty(llr, bit)
                    new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_paths = [
            p
            for p in paths
            if self.crc_length == 0
            or crc_check(p.u_hat[self._info_idx], self.crc_length)
        ]
        pool = crc_paths if crc_paths else paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
