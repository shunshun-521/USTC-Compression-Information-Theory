"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import _MaunderSC, _pm_penalty, sc_decode
from encoder import bit_reversal_permutation

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    def __init__(self, N, llr_brp, frozen_bits):
        self.N = N
        self.frozen_bits = frozen_bits
        self.sc = _MaunderSC(N)
        self.sc.llrs[:, self.sc.n] = llr_brp
        self.sc.llrs_updated.fill(False)
        self.sc.bits_updated.fill(False)
        self.sc.bits_updated[:, 0] = frozen_bits == 1
        self.sc.llrs_updated[:, self.sc.n] = True
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)

    def clone(self):
        return copy.deepcopy(self)


class SCLDecoder:
    """SCL 译码器（路径级复制 Maunder SC 状态）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        brp = bit_reversal_permutation(self.N)
        llr_brp = llr_ch[brp]
        paths = [_Path(self.N, llr_brp, self.frozen_bits)]

        for i in range(self.N):
            new_paths = []
            for p in paths:
                p.sc._update_llr(i, 0)
                llr_i = p.sc.llrs[i, 0]
                if self.frozen_bits[i]:
                    q = p.clone()
                    q.pm += _pm_penalty(llr_i, 0)
                    q.u[i] = 0
                    q.sc.bits[i, 0] = 0
                    q.sc.bits_updated[i, 0] = True
                    new_paths.append(q)
                else:
                    for bit in (0, 1):
                        q = p.clone()
                        q.pm += _pm_penalty(llr_i, bit)
                        q.u[i] = bit
                        q.sc.bits[i, 0] = bit
                        q.sc.bits_updated[i, 0] = True
                        new_paths.append(q)
            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u[self.info_positions], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda x: x.pm)
        return best.u, best.pm


def scl_equivalent_to_sc(N, frozen_bits, llr_ch):
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
