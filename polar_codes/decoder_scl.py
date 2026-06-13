"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import bit_reversed, _update_llrs, _update_bits, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


def _tree_to_nat(raw, n):
    rev = np.array([bit_reversed(i, n) for i in range(len(raw))], dtype=int)
    return raw[rev]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _is_frozen(self, tree_idx):
        nat = bit_reversed(tree_idx, self.n)
        return self.frozen_bits[nat] == 1

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []
            for pidx, path in enumerate(paths):
                _update_llrs(path["L"], path["B"], l, self.n)
                llr = path["L"][l, self.n]
                if self._is_frozen(l):
                    candidates.append((_pm_update(path["pm"], llr, 0), pidx, 0))
                else:
                    for u in (0, 1):
                        candidates.append((_pm_update(path["pm"], llr, u), pidx, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for new_pm, parent_idx, u_val in candidates:
                parent = paths[parent_idx]
                child = {
                    "pm": new_pm,
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                }
                child["B"][l, self.n] = u_val
                _update_bits(child["B"], l, self.n, self.N)
                new_paths.append(child)
            paths = new_paths

        for path in paths:
            u_hat = _tree_to_nat(path["B"][:, self.n].astype(int), self.n)
            u_hat[self.frozen_bits == 1] = 0
            path["u_hat"] = u_hat

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_positions], self.crc_length)
            ]
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
