"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
    precompute_sc_indices,
    _frozen_set_from_mask,
    sc_decode,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = bits[-crc_length:]
    actual = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.array_equal(actual, expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        _, self.llr_layers, self.bit_layers = precompute_sc_indices(N)

    def _update_llrs(self, path, phi):
        l = _bit_reversed(phi, self.n)
        for s in self.llr_layers[phi]:
            bs = 2 ** (s + 1)
            br = bs // 2
            for j in range(l, self.N, bs):
                if j % bs < br:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + br, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - br, s], path.B[j - br, s + 1]
                    )

    def _update_bits(self, path, phi):
        l = _bit_reversed(phi, self.n)
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            bs = int(2 ** s)
            br = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= br:
                    path.B[j - br, s - 1] = int(path.B[j, s]) ^ int(path.B[j - br, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            mask = np.zeros(self.N, dtype=int)
            for idx in self.frozen_set:
                mask[idx] = 1
            u_hat = sc_decode(llr_ch, mask)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch[self.rev]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, phi)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr_bit, 0)
                    path.u_hat[phi] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, phi)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = _Path(self.N, self.n)
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.pm = path.pm + self._pm_penalty(llr_bit, bit)
                        p.u_hat = path.u_hat.copy()
                        p.u_hat[phi] = bit
                        p.B[l, self.n] = bit
                        self._update_bits(p, phi)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                info_mask = np.ones(self.N, dtype=bool)
                for idx in self.frozen_set:
                    info_mask[idx] = False
                info_bits = p.u_hat[info_mask]
                crc_ok.append(crc_check(info_bits, self.crc_length))
            else:
                crc_ok.append(True)

        if any(crc_ok):
            candidates = [p for p, ok in zip(paths, crc_ok) if ok]
        else:
            candidates = paths

        best = min(candidates, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
