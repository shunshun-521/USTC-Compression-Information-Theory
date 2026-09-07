"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _leaf_llr
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            msb = (reg >> (crc_length - 1)) & 1
            reg = (reg << 1) & ((1 << crc_length) - 1)
            if msb:
                reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -50.0, 50.0)
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=np.int8)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = _leaf_llr(llr_ch, path["u"], phi)
                if self.frozen_bits[phi]:
                    new_u = path["u"].copy()
                    new_u[phi] = 0
                    candidates.append(
                        {"pm": path["pm"] + self._penalty(llr_phi, 0), "u": new_u}
                    )
                else:
                    for bit in (0, 1):
                        new_u = path["u"].copy()
                        new_u[phi] = bit
                        candidates.append(
                            {"pm": path["pm"] + self._penalty(llr_phi, bit), "u": new_u}
                        )
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best_crc = None
        best_all = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            for path in paths:
                info_bits = path["u"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path

        chosen = best_crc if best_crc is not None else best_all
        return chosen["u"].copy(), chosen["pm"]
