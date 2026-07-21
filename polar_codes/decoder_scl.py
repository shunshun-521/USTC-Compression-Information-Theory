"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_llr_at_phi


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


def _pm_add(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    if bit != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array([i for i in range(N) if i not in self.frozen_set], dtype=int)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "bits": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = sc_llr_at_phi(llr_ch, path["bits"], phi, self.frozen_bits)
                if phi in self.frozen_set:
                    bits = path["bits"].copy()
                    bits[phi] = 0
                    pm = _pm_add(path["pm"], llr, 0)
                    candidates.append((pm, bits))
                else:
                    for bit in (0, 1):
                        bits = path["bits"].copy()
                        bits[phi] = bit
                        pm = _pm_add(path["pm"], llr, bit)
                        candidates.append((pm, bits))

            candidates.sort(key=lambda x: x[0])
            paths = [{"pm": pm, "bits": bits} for pm, bits in candidates[: self.list_size]]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["bits"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["bits"], best["pm"]
