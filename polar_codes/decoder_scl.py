"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _permute_llr_for_decode,
    _update_bits,
    _update_llrs,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = _crc_remainder(payload, poly, crc_length)
    actual = 0
    for bit in bits[-crc_length:]:
        actual = (actual << 1) | int(bit)
    return expected == actual


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        n = self.L.shape[1] - 1
        N = self.L.shape[0]
        new_path = Path(n, N)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        precompute_sc_indices(N)

    def _path_metric_penalty(self, llr, u_bit):
        preferred = 0 if llr >= 0 else 1
        return 0.0 if u_bit == preferred else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _permute_llr_for_decode(llr_ch)
        N = self.N
        n = self.n

        active = [Path(n, N)]
        active[0].L[:, 0] = llr_ch.copy()

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []

            for path in active:
                _update_llrs(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += self._path_metric_penalty(llr, 0)
                    new_path.B[l, n] = 0
                    new_path.u_hat[l] = 0
                    _update_bits(new_path.B, l, n, N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr, u_bit)
                        new_path.B[l, n] = u_bit
                        new_path.u_hat[l] = u_bit
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            active = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in active
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else active, key=lambda p: p.pm)
        else:
            best = min(active, key=lambda p: p.pm)

        return best.u_hat, best.pm
