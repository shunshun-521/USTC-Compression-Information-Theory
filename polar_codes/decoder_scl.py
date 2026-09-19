"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import _bitrev_indices

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（系统型长除法）。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    for i in range(len(info_bits)):
        if msg[i]:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    msg[i + j] ^= 1
    return np.concatenate([info_bits, msg[-crc_length:]])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确（LFSR 余数为零）。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class _Path:
    """单条译码路径。"""

    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_init

    def copy(self):
        new = _Path.__new__(_Path)
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        return new


class SCLDecoder:
    """SCL 译码器（置换 SC + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_init = llr_ch[_bitrev_indices(self.N)]

        paths = [_Path(self.N, self.n, llr_init)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = path.copy()
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

            for path in paths:
                _update_bits(path.B, l, self.n, self.N)

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_pass(p)]
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm

    def _crc_pass(self, path):
        info_bits = path.B[:, self.n].astype(int)[self.info_indices]
        return crc_check(info_bits, self.crc_length)
