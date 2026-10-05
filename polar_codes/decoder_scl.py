"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _hard_decision,
    _lower_llr,
    _update_bits,
    _update_llrs,
    _upper_llr,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc8_serial(bits):
    reg = 0
    for b in bits:
        reg ^= int(b) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ _CRC8_POLY) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return reg


def _crc16_serial(bits):
    reg = 0
    for b in bits:
        reg ^= int(b) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        rem = _crc8_serial(info_bits)
        crc_bits = np.array([(rem >> i) & 1 for i in range(7, -1, -1)], dtype=np.int8)
    elif crc_length == 16:
        rem = _crc16_serial(info_bits)
        crc_bits = np.array([(rem >> i) & 1 for i in range(15, -1, -1)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _pm_add(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if self.frozen_bits[l]:
                    child = self._clone(path)
                    child.pm += self._pm_add(llr, 0)
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = self._clone(path)
                        child.pm += self._pm_add(llr, u)
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    return p.u_hat.copy(), p.pm
        best = paths[0]
        return best.u_hat.copy(), best.pm

    def _clone(self, path):
        q = _Path(self.N, self.n)
        q.L = path.L.copy()
        q.B = path.B.copy()
        q.pm = path.pm
        q.u_hat = path.u_hat.copy()
        return q
