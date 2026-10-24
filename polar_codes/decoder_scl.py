"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_decode, sc_phase_llr


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    padded = list(info_bits) + [0] * crc_length
    rem = _crc_remainder(padded, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=np.int64) if info_indices is not None else None
        )

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [(0.0, np.zeros(self.N, dtype=np.int8))]

        for phi in range(self.N):
            candidates = []
            for pm, dec in paths:
                llr = sc_phase_llr(llr_ch, self.frozen_bits, dec, phi)
                if self.frozen_bits[phi]:
                    d = dec.copy()
                    d[phi] = 0
                    candidates.append((pm + self._pm_penalty(llr, 0), d))
                else:
                    for u in (0, 1):
                        d = dec.copy()
                        d[phi] = u
                        candidates.append((pm + self._pm_penalty(llr, u), d))
            candidates.sort(key=lambda x: x[0])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for pm, d in paths:
                msg = d[self.info_indices] if self.info_indices is not None else d
                if crc_check(msg, self.crc_length):
                    valid.append((pm, d))
            if valid:
                pm, dec = min(valid, key=lambda x: x[0])
                return dec.astype(int), pm

        pm, dec = min(paths, key=lambda x: x[0])
        return dec.astype(int), pm
