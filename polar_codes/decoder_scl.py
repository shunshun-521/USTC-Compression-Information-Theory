"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode_recursive, sc_llr_at_phase, align_llr_for_decoder


# ==================== CRC 工具 ====================

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
    """计算 CRC 并附加到信息比特末尾。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class SCLDecoder:
    """SCL 译码器（串行抵消列表，路径度量 PM 越小越好）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=np.int64)

    def decode(self, llr_ch):
        llr_ch = align_llr_for_decoder(llr_ch)

        if self.list_size == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=np.int8)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_phi = sc_llr_at_phase(llr_ch, self.frozen_bits, path["u"][:phi], phi)
                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr_phi >= 0 else abs(llr_phi)
                    p = path.copy()
                    p["pm"] = path["pm"] + penalty
                    p["u"] = path["u"].copy()
                    p["u"][phi] = 0
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        p["u"] = path["u"].copy()
                        p["u"][phi] = bit
                        if (bit == 0 and llr_phi < 0) or (bit == 1 and llr_phi >= 0):
                            p["pm"] = path["pm"] + abs(llr_phi)
                        else:
                            p["pm"] = path["pm"]
                        new_paths.append(p)

            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda x: x["pm"])
        if self.crc_length > 0:
            ok_paths = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            if ok_paths:
                best = min(ok_paths, key=lambda x: x["pm"])
        return best["u"].copy(), float(best["pm"])
