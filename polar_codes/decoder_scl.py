"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _upper_llr,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc8_remainder(bits):
    crc = 0
    for b in bits:
        crc ^= int(b) << 7
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ _CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_remainder(bits):
    reg = 0
    for b in bits:
        reg ^= int(b) << 15
        for _ in range(8):
            if reg & 0x8000:
                reg = ((reg << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位（CRC-8: 0x07, CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        rem = _crc8_remainder(info_bits)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        rem = _crc16_remainder(info_bits)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        return _crc8_remainder(bits) == 0
    if crc_length == 16:
        return _crc16_remainder(bits) == 0
    raise ValueError("crc_length must be 8 or 16")


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制在分裂时进行，列表规模适中时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen = set(np.where(frozen_bits.astype(bool))[0])
        self.info_indices = np.where(~frozen_bits.astype(bool))[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    btm = path.L[j, s]
                    top = path.L[j - branch_size, s]
                    b = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = btm + top if b == 0 else btm - top

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br]

        paths = [_Path(self.N, self.n, llr.copy())]
        order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in order:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]
                if l in self.frozen:
                    pen = self._pm_penalty(cur_llr, 0)
                    path.pm += pen
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(self.N, self.n, path.L[:, 0].copy())
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(cur_llr, u_bit)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = u_bit
                        child.B[l, self.n] = u_bit
                        self._update_bits(child, l)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = paths
        if self.crc_length > 0:
            passed = []
            for p in candidates:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(p)
            if passed:
                candidates = passed
        best = min(candidates, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
