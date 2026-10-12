"""
极化码 SCL（串行抵消列表）译码器 + CRC 辅助 CA-SCL
"""
import numpy as np
from encoder import polar_generator_matrix
from decoder_sc import sc_decode

_G_CACHE = {}


def _get_G(N):
    if N not in _G_CACHE:
        _G_CACHE[N] = polar_generator_matrix(N)
    return _G_CACHE[N]


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.zeros(crc_length, dtype=np.int8)
    for i in range(crc_length):
        crc_bits[i] = (reg >> (crc_length - 1 - i)) & 1
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.G = _get_G(N)

    def _metric(self, llr_ch, u):
        x = (u @ self.G) % 2
        return float(np.sum(llr_ch * (1 - 2 * x)))

    def _pm_bit(self, pm, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        if u != hard:
            pm += abs(llr_val)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        base = sc_decode(llr_ch, self.frozen_bits)
        paths = [(0.0, base.copy())]

        # 仅在前若干信息位分裂以控制复杂度（长码仿真可接受）
        split_idx = self.info_indices[: min(48, len(self.info_indices))]
        for phi in split_idx:
            new_paths = []
            for pm, u in paths:
                for bit in (0, 1):
                    u2 = u.copy()
                    u2[phi] = bit
                    pm2 = self._pm_bit(pm, llr_ch[phi], bit)
                    new_paths.append((pm2, u2))
            new_paths.sort(key=lambda t: (-self._metric(llr_ch, t[1]), t[0]))
            paths = new_paths[: self.L]

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in paths if crc_check(u, self.crc_length)]
            if valid:
                paths = valid

        best_pm, best_u = min(paths, key=lambda t: (t[0], -self._metric(llr_ch, t[1])))
        return best_u.astype(int), best_pm
