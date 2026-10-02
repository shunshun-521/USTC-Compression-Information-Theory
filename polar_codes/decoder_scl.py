"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _update_llrs,
    _update_bits,
    f_operation,
    g_operation,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg <<= 1
        reg |= int(b)
        reg &= (1 << crc_length) - 1
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _crc_remainder(padded, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("pm", "L", "B", "x_hat", "active")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.x_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._decode_order = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, u_bit):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for l in self._decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    path.x_hat[l] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p = _Path(self.n, self.N)
                        p.pm = path.pm
                        p.L[:] = path.L
                        p.B[:] = path.B
                        p.x_hat[:] = path.x_hat
                        p.pm += self._pm_penalty(llr, u_bit)
                        p.B[l, self.n] = u_bit
                        p.x_hat[l] = u_bit
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.x_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.x_hat.copy(), best.pm
