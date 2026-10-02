"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    f_operation,
    g_operation,
    _llr_to_decode_order,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_run(bits):
    crc = 0
    for b in bits:
        crc ^= int(b) << 7
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_run(bits):
    crc = 0
    for b in bits:
        crc ^= int(b) << 15
        for _ in range(16):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """附加 CRC 校验位（MSB 优先 LFSR，多项式 0x07 / 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        reg = _crc8_run(info_bits)
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        reg = _crc16_run(info_bits)
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验：信息位与 CRC 位整体输入 LFSR 后余数为零"""
    bits = np.asarray(bits, dtype=int).ravel()
    info = bits[:-crc_length]
    crc = bits[-crc_length:]
    if crc_length == 8:
        expected = np.array([(_crc8_run(info) >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        expected = np.array([(_crc16_run(info) >> (15 - i)) & 1 for i in range(16)], dtype=int)
    return np.array_equal(crc, expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], int(top_bit)
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2**s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr = _llr_to_decode_order(np.asarray(llr_ch, dtype=np.float64))
        paths = [_Path(self.N, self.n, llr)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_l = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr_l, 0)
                    path.B[l, self.n] = 0
                    path.u[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p0 = _Path(self.N, self.n, llr)
                        p0.L = path.L.copy()
                        p0.B = path.B.copy()
                        p0.pm = path.pm
                        p0.u = path.u.copy()
                        p0.pm += self._pm_penalty(llr_l, u_bit)
                        p0.B[l, self.n] = u_bit
                        p0.u[l] = u_bit
                        self._update_bits(p0, l)
                        new_paths.append(p0)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [p for p in paths if crc_check(p.u[info_idx], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u.copy(), best.pm
