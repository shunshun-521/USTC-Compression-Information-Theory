"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import channel_llr_to_decode_domain, f_operation, g_operation, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ bit) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) | fb
        if fb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ bit) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) | fb
        if fb:
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """递归列表 SC 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def _scl_rec(self, llr, indices, paths):
        if len(indices) == 1:
            i = indices[0]
            new_paths = []
            for pm, dec in paths:
                if self.frozen_bits[i]:
                    bit = 0
                    pm2 = self._pm_update(pm, llr[0], 0)
                else:
                    for bit in (0, 1):
                        pm2 = self._pm_update(pm, llr[0], bit)
                        d = dec.copy()
                        d[i] = bit
                        new_paths.append((pm2, d))
                    continue
                d = dec.copy()
                d[i] = bit
                new_paths.append((pm2, d))
            new_paths.sort(key=lambda x: x[0])
            return new_paths[: self.list_size]

        half = len(indices) // 2
        left_llr = f_operation(llr[:half], llr[half:])
        paths = self._scl_rec(left_llr, indices[:half], paths)
        expanded = []
        for pm, dec in paths:
            left_bits = np.array([dec[i] for i in indices[:half]], dtype=int)
            right_llr = g_operation(llr[:half], llr[half:], left_bits)
            expanded.append((pm, dec, right_llr))
        merged = []
        for pm, dec, right_llr in expanded:
            sub = self._scl_rec(right_llr, indices[half:], [(pm, dec)])
            merged.extend(sub)
        merged.sort(key=lambda x: x[0])
        return merged[: self.list_size]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_dec = channel_llr_to_decode_domain(llr_ch)
        paths = self._scl_rec(llr_dec, list(range(self.N)), [(0.0, {})])
        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            valid = []
            for pm, dec in paths:
                u = np.zeros(self.N, dtype=int)
                for i, v in dec.items():
                    u[i] = v
                if crc_check(u[info_pos], self.crc_length):
                    valid.append((pm, dec))
            if valid:
                paths = valid
        pm, dec = min(paths, key=lambda x: x[0])
        u_hat = np.zeros(self.N, dtype=int)
        for i, v in dec.items():
            u_hat[i] = v
        return u_hat, pm
