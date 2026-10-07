"""
极化码 SCL（串行抵消列表）译码器 + CA-SCL
列表在码字域按 LLR 可信度翻转生成候选，再映射回 u（与 F^{⊗n} 编码配套）
"""
import numpy as np
import math

from gf2 import F2, gf2_inv
from decoder_sc import _hard_gi_decode


CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_process(bits, crc_length):
    poly = CRC_POLY[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    for b in np.asarray(bits, dtype=np.int8).ravel():
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if msb ^ int(b):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    reg = _crc_process(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) < crc_length:
        return False
    return _crc_process(bits, crc_length) == 0


def _path_metric(llr_ch, u, G):
    x = (u @ G) % 2
    x_bpsk = 1.0 - 2.0 * x
    return float(np.sum((llr_ch - x_bpsk) ** 2))


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.G = F2(self.n)
        self.Gi = gf2_inv(self.G)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        x_hat = (llr_ch < 0).astype(int)
        candidates = [x_hat.copy()]
        order = np.argsort(np.abs(llr_ch))
        for idx in order:
            if len(candidates) >= self.list_size:
                break
            for c in list(candidates):
                x2 = c.copy()
                x2[idx] ^= 1
                candidates.append(x2)
                if len(candidates) >= self.list_size:
                    break

        best_u = None
        best_pm = np.inf
        for x_c in candidates[: self.list_size]:
            u_c = (x_c @ self.Gi) % 2
            u_c[self.frozen_bits] = 0
            pm = _path_metric(llr_ch, u_c, self.G)
            if pm < best_pm:
                best_pm = pm
                best_u = u_c

        if self.crc_length > 0 and best_u is not None:
            valid = []
            for x_c in candidates[: self.list_size]:
                u_c = (x_c @ self.Gi) % 2
                u_c[self.frozen_bits] = 0
                if crc_check(u_c, self.crc_length):
                    valid.append((u_c, _path_metric(llr_ch, u_c, self.G)))
            if valid:
                best_u, best_pm = min(valid, key=lambda t: t[1])

        if best_u is None:
            best_u = _hard_gi_decode(llr_ch, self.frozen_bits)
            best_pm = _path_metric(llr_ch, best_u, self.G)
        return best_u.astype(int), best_pm
