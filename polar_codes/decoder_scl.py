"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import _SCD, bit_reversed, sc_decode
from encoder import bit_reversal_permutation


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _compute_crc_bits(info_bits, crc_length=8):
    poly = CRC_POLYS[crc_length]
    crc = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        crc ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if crc & (1 << (crc_length - 1)):
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return np.array([(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    return np.concatenate([info_bits, _compute_crc_bits(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], _compute_crc_bits(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（路径分裂时深拷贝 SCD 状态）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br_inv = np.argsort(bit_reversal_permutation(N))

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self.br_inv]
        paths = [{"scd": _SCD(self.N, self.n, llr_perm.copy()), "pm": 0.0}]

        for phi in [bit_reversed(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                scd = path["scd"]
                scd._update_llrs(phi)
                llr = scd.L[phi, self.n]

                if phi in self.frozen_indices:
                    child = {"scd": copy.deepcopy(scd), "pm": path["pm"] + self._pm_penalty(llr, 0)}
                    child["scd"].B[phi, self.n] = 0
                    child["scd"]._update_bits(phi)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = {"scd": copy.deepcopy(scd), "pm": path["pm"] + self._pm_penalty(llr, bit)}
                        child["scd"].B[phi, self.n] = bit
                        child["scd"]._update_bits(phi)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = path["scd"].B[:, self.n].astype(int)
            candidates.append((path["pm"], u_hat))

        if self.crc_length > 0:
            crc_ok = [
                (pm, u) for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if crc_ok:
                candidates = crc_ok

        pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, pm
