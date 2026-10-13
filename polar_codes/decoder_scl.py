"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode, sc_decode_recursive

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器：按比特顺序扩展路径（Lazy Copy 为路径数组复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _llr_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _bit_llr(self, llr_ch, known_bits, phi):
        u_tmp = sc_decode_recursive(llr_ch, self.frozen_bits, known_bits=known_bits)
        # 近似：用完整 SC 在 phi 处的判决反推 LLR 符号
        bit = u_tmp[phi]
        # 通过两次已知前缀译码估计边际（高 SNR 下足够）
        known0 = dict(known_bits)
        known0[phi] = 0
        known1 = dict(known_bits)
        known1[phi] = 1
        u0 = sc_decode_recursive(llr_ch, self.frozen_bits, known_bits=known0)[phi]
        u1 = sc_decode_recursive(llr_ch, self.frozen_bits, known_bits=known1)[phi]
        if u0 == 0 and u1 == 0:
            return 1000.0
        if u0 == 1 and u1 == 1:
            return -1000.0
        return 1000.0 if bit == 0 else -1000.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [{"pm": 0.0, "known": {}}]
        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr0 = self._bit_llr(llr_ch, path["known"], phi)
                if self.frozen_bits[phi]:
                    known = dict(path["known"])
                    known[phi] = 0
                    new_paths.append({"pm": path["pm"] + self._llr_penalty(llr0, 0), "known": known})
                else:
                    for u in (0, 1):
                        known = dict(path["known"])
                        known[phi] = u
                        new_paths.append(
                            {"pm": path["pm"] + self._llr_penalty(llr0, u), "known": known}
                        )
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L]

        best_crc = None
        best_pm = float("inf")
        best_any = paths[0]
        for path in paths:
            if path["pm"] < best_any["pm"]:
                best_any = path
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits, known_bits=path["known"])
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length) and path["pm"] < best_pm:
                    best_pm = path["pm"]
                    best_crc = (u_hat, path["pm"])
        if best_crc is not None:
            return best_crc
        u_hat = sc_decode_recursive(llr_ch, self.frozen_bits, known_bits=best_any["known"])
        return u_hat, best_any["pm"]
