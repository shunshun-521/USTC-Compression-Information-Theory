"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _SCD, _bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_step(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCD(self.N, llr_ch, self.frozen_set)]
        paths[0].pm = 0.0

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                path._update_llrs(l)
                llr = path.L[l, path.n]

                if l in self.frozen_set:
                    new_path = path.copy_state()
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.B[l, path.n] = 0
                    new_path._update_bits(l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy_state()
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.B[l, path.n] = bit
                        new_path._update_bits(l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.B[:, path.n].astype(np.int8)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        u_hat = best.B[:, best.n].astype(np.int8)
        return u_hat.copy(), best.pm
