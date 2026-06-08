"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import sc_decode, sc_llr_at_phase


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


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
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.llr_base = None

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _crc_ok(self, u_hat):
        info_positions = np.where(self.frozen_bits == 0)[0]
        payload = u_hat[info_positions]
        return crc_check(payload, self.crc_length)

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [(0.0, np.zeros(self.N, dtype=int))]
        frozen_bool = self.frozen_bits.astype(bool)

        for phi in range(self.N):
            candidates = []
            for pm, u_hat in paths:
                llr_bit = sc_llr_at_phase(llr_ch, frozen_bool, u_hat, phi)
                if self.frozen_bits[phi]:
                    new_pm = pm + self._pm_penalty(llr_bit, 0)
                    child = u_hat.copy()
                    child[phi] = 0
                    candidates.append((new_pm, child))
                else:
                    for u in (0, 1):
                        new_pm = pm + self._pm_penalty(llr_bit, u)
                        child = u_hat.copy()
                        child[phi] = u
                        candidates.append((new_pm, child))

            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [(pm, u) for pm, u in paths if self._crc_ok(u)]
            pool = crc_pass if crc_pass else paths
        else:
            pool = paths

        best_pm, best_u = min(pool, key=lambda x: x[0])
        return best_u, best_pm
