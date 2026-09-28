"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import llr_at_phase, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


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
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
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

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr = llr_at_phase(llr_ch, self.frozen_bits, path["u"], phi)
                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    cand = {"pm": path["pm"] + penalty, "u": path["u"].copy()}
                    cand["u"][phi] = 0
                    new_paths.append(cand)
                else:
                    for u_bit in (0, 1):
                        penalty = 0.0 if (u_bit == 0 and llr >= 0) or (u_bit == 1 and llr < 0) else abs(llr)
                        cand = {"pm": path["pm"] + penalty, "u": path["u"].copy()}
                        cand["u"][phi] = u_bit
                        new_paths.append(cand)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            for path in paths:
                info_bits = path["u"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return path["u"], path["pm"]
        best = paths[0]
        return best["u"], best["pm"]


def scl_equals_sc_test(N=64, K=32):
    """L=1 时应等价于 SC"""
    from construction import ga_construction
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(10):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = rng.normal(0, 2, N)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
