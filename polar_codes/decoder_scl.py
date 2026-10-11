"""
极化码 SCL（串行抵消列表）译码器 — 基于码字 Chase 列表 + GF(2) 源恢复
"""
import numpy as np

from decoder_sc import _gf2_solve, _get_generator

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process_bits(bits, crc_length, poly):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(b)) & mask
        if msb ^ int(b):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    extended = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    reg = _crc_process_bits(extended, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process_bits(bits, crc_length, poly) == 0


def _path_metric(llr_ch, x):
    """对数域路径度量（越小越好）：sum log(1+exp(-sign*llr))."""
    s = (1 - 2 * x) * llr_ch
    return float(np.sum(np.log1p(np.exp(-np.abs(s)))))


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]
        self.G = _get_generator(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        x0 = (llr_ch < 0).astype(int)
        reliability = np.argsort(np.abs(llr_ch))

        candidates = {tuple(x0): _path_metric(llr_ch, x0)}
        depth = min(self.list_size, self.N)
        for idx in reliability[:depth]:
            new_cands = {}
            for x_tuple, pm in list(candidates.items()):
                x = np.array(x_tuple, dtype=int)
                for bit in (0, 1):
                    x_try = x.copy()
                    x_try[idx] = bit
                    key = tuple(x_try)
                    pm_try = _path_metric(llr_ch, x_try)
                    if key not in new_cands or pm_try < new_cands[key]:
                        new_cands[key] = pm_try
            candidates = dict(
                sorted(new_cands.items(), key=lambda kv: kv[1])[: self.list_size]
            )

        paths = []
        for x_tuple, pm in candidates.items():
            x = np.array(x_tuple, dtype=int)
            u = _gf2_solve(self.G.T, x)
            u[self.frozen_bits == 1] = 0
            paths.append((pm, u))

        paths.sort(key=lambda t: t[0])
        if self.crc_length > 0:
            for pm, u in paths:
                info_bits = u[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    return u.copy(), pm
        pm, u = paths[0]
        return u.copy(), pm
