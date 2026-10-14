"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_bp import BPDecoder
from decoder_sc import sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:]
    )


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（BP 辅助逐位 LLR + 路径分裂）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
        self._bp = BPDecoder(N, frozen_bits, max_iter=1)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [(0.0, {})]

        for phi in range(self.N):
            new_paths = []
            for pm, prior in paths:
                llr_phi = self._bp.bit_marginal(llr_ch, prior, phi)
                if self.frozen_bits[phi]:
                    prior2 = dict(prior)
                    prior2[phi] = 0
                    new_paths.append((pm + _pm_penalty(llr_phi, 0), prior2))
                else:
                    for bit in (0, 1):
                        prior2 = dict(prior)
                        prior2[phi] = bit
                        new_paths.append((pm + _pm_penalty(llr_phi, bit), prior2))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        u_hat = np.zeros(self.N, dtype=int)
        best_pm, best_prior = paths[0]
        for idx, bit in best_prior.items():
            u_hat[idx] = bit
        for phi in range(self.N):
            if self.frozen_bits[phi]:
                u_hat[phi] = 0

        if self.crc_length > 0:
            candidates = []
            for pm, prior in paths:
                u = np.zeros(self.N, dtype=int)
                for idx, bit in prior.items():
                    u[idx] = bit
                for phi in range(self.N):
                    if self.frozen_bits[phi]:
                        u[phi] = 0
                if crc_check(u[self.info_positions], self.crc_length):
                    candidates.append((pm, u))
            if candidates:
                best_pm, u_hat = min(candidates, key=lambda x: x[0])

        return u_hat, best_pm


def scl_equivalent_to_sc(N, frozen_bits, llr_ch):
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = scl.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
