"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
基于 lazy SC 结构扩展列表译码
"""
import numpy as np
import math

from decoder_sc import f_operation, g_operation, _permute_channel_llr, sc_decode

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for b in bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return reg == 0
    else:
        reg = 0
        for b in bits:
            reg ^= int(b) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        return reg == 0


class _LazyPath:
    """单条 SCL 路径，复用 lazy SC 的 LLR/比特递推结构。"""

    __slots__ = ('pm', 'llrs', 's', 'N', 'n')

    def __init__(self, N, n, llr_channel):
        self.pm = 0.0
        self.llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
        self.llrs[n, :] = llr_channel
        self.s = np.full((n + 1, N), -1, dtype=np.int8)
        self.N = N
        self.n = n

    def copy(self):
        p = _LazyPath.__new__(_LazyPath)
        p.pm = self.pm
        p.llrs = self.llrs.copy()
        p.s = self.s.copy()
        p.N = self.N
        p.n = self.n
        return p

    def _b_check(self, ll, ii):
        return (ii // (1 << ll)) % 2

    def _s_updater(self, ll, ii):
        if self._b_check(ll - 1, ii):
            self.s[ll, ii] = self.s[ll - 1, ii]
        else:
            if self.s[ll - 1, ii] == -1:
                self._s_updater(ll - 1, ii)
            j = ii + (1 << (ll - 1))
            if self.s[ll - 1, j] == -1:
                self._s_updater(ll - 1, j)
            self.s[ll, ii] = self.s[ll - 1, ii] ^ self.s[ll - 1, j]

    def _li(self, ll, ii):
        if self.llrs[ll, ii] != -np.inf:
            return self.llrs[ll, ii]
        if self._b_check(ll, ii) == 0:
            self.llrs[ll, ii] = f_operation(
                self._li(ll + 1, ii), self._li(ll + 1, ii + (1 << ll))
            )
        else:
            if ll > 0:
                self._s_updater(ll, ii - (1 << ll))
            self.llrs[ll, ii] = g_operation(
                self._li(ll + 1, ii - (1 << ll)),
                self._li(ll + 1, ii),
                self.s[ll, ii - (1 << ll)],
            )
        return self.llrs[ll, ii]

    def u_hat(self):
        return self.s[0, :].astype(int)


class SCLDecoder:
    """SCL 译码器（lazy SC 扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        Lsz = self.list_size
        frozen = self.frozen_bits

        llr_nat = _permute_channel_llr(llr_ch, N)
        paths = [_LazyPath(N, n, llr_nat)]

        for ii in range(N):
            candidates = []
            for path in paths:
                cur_llr = path._li(0, ii)
                if frozen[ii]:
                    pen = abs(cur_llr) if cur_llr < 0 else 0.0
                    path.pm += pen
                    path.s[0, ii] = 0
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        pen = 0.0
                        if bit == 0 and cur_llr < 0:
                            pen = abs(cur_llr)
                        elif bit == 1 and cur_llr >= 0:
                            pen = abs(cur_llr)
                        new_path.pm += pen
                        new_path.s[0, ii] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:Lsz]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_dec = p.u_hat()
                info_bits = u_dec[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat(), best.pm
