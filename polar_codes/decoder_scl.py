"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _bit_llr


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位（MSB-first）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（路径复制 + 路径度量裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u == u_hard:
            return pm
        return pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        if self.list_size == 1:
            from decoder_sc import sc_decode

            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(N, dtype=np.int8)}]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr0 = _bit_llr(llr_ch, path["u"], phi, 0)
                if self.frozen_bits[phi]:
                    pm = self._pm_update(path["pm"], llr0, 0)
                    u_new = path["u"].copy()
                    u_new[phi] = 0
                    new_paths.append({"pm": pm, "u": u_new})
                else:
                    for u_bit in (0, 1):
                        pm = self._pm_update(path["pm"], llr0, u_bit)
                        u_new = path["u"].copy()
                        u_new[phi] = u_bit
                        new_paths.append({"pm": pm, "u": u_new})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_ok(p["u"])]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"].copy(), best["pm"]

    def _crc_ok(self, u_hat):
        info_bits = u_hat[self.info_positions]
        return crc_check(info_bits, self.crc_length)
