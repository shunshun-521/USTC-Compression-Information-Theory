"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import itertools

import numpy as np

from decoder_sc import sc_decode


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _path_metric(llr_ch, u_hat, frozen_bits):
    pm = 0.0
    for i, b in enumerate(u_hat):
        if frozen_bits[i]:
            continue
        hard = 0 if llr_ch[i] >= 0 else 1
        if b != hard:
            pm += abs(llr_ch[i])
    return pm


def _sc_with_prefix(llr_ch, frozen_bits, prefix):
    llr = llr_ch.astype(np.float64).copy()
    fr = np.asarray(frozen_bits, dtype=bool).copy()
    for idx, val in prefix.items():
        fr[idx] = True
        llr[idx] = 50.0 if val == 0 else -50.0
    return sc_decode(llr, fr)


class SCLDecoder:
    """SCL：在前若干信息位上枚举路径，其余位用 SC 延续。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        m = int(np.round(np.log2(self.list_size)))
        m = max(1, min(m, len(self.info_idx)))
        branch_idx = self.info_idx[:m]

        candidates = []
        for pattern in itertools.product([0, 1], repeat=m):
            prefix = {int(branch_idx[i]): int(pattern[i]) for i in range(m)}
            u_hat = _sc_with_prefix(llr_ch, self.frozen_bits, prefix)
            pm = _path_metric(llr_ch, u_hat, self.frozen_bits)
            candidates.append((pm, u_hat))

        candidates.sort(key=lambda x: x[0])
        candidates = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for pm, uh in candidates:
                bits = uh[self.info_idx]
                if crc_check(bits, self.crc_length):
                    valid.append((pm, uh))
            if valid:
                candidates = valid

        pm, u_hat = candidates[0]
        return u_hat.astype(int), pm
