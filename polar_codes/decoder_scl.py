"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import sc_decode, llr_at_bit_phi, LLR_MAX


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in np.asarray(info_bits, dtype=np.int8):
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([np.asarray(info_bits, dtype=np.int8), crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:]
    )


def _path_metric_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        return pm + abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（逐比特路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_ind = np.zeros(N, dtype=np.int8)
        self.frozen_ind[self.frozen_bits] = 1
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr = np.clip(llr_ch[br], -LLR_MAX, LLR_MAX)

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_phi = llr_at_bit_phi(llr, self.frozen_ind, path["u"], phi)
                if self.frozen_ind[phi]:
                    npm = _path_metric_update(path["pm"], llr_phi, 0)
                    q = {"pm": npm, "u": path["u"].copy()}
                    q["u"][phi] = 0
                    new_paths.append(q)
                else:
                    for bit in (0, 1):
                        npm = _path_metric_update(path["pm"], llr_phi, bit)
                        q = {"pm": npm, "u": path["u"].copy()}
                        q["u"][phi] = bit
                        new_paths.append(q)
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        best = None
        if self.crc_length > 0:
            for p in paths:
                info_bits = p["u"][self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    if best is None or p["pm"] < best["pm"]:
                        best = p
        if best is None:
            best = min(paths, key=lambda x: x["pm"])
        return best["u"].copy(), float(best["pm"])
