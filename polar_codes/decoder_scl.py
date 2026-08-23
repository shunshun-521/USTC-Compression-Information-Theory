"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _load_ref_modules,
    g_operation,
    f_operation,
    sc_decode,
)
from encoder import bit_reversal_permutation

CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _crc_update_msb(crc, bit, poly, crc_length):
    inv = ((crc >> (crc_length - 1)) ^ int(bit)) & 1
    crc = ((crc << 1) & ((1 << crc_length) - 1)) ^ (poly if inv else 0)
    return crc


def _crc_remainder(info_bits, poly, crc_length):
    crc = 0
    for bit in info_bits:
        crc = _crc_update_msb(crc, bit, poly, crc_length)
    return np.array([(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    crc = 0
    for bit in bits:
        crc = _crc_update_msb(crc, bit, poly, crc_length)
    return crc == 0


def _path_metric_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    return pm if u == u_hard else pm + abs(llr)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]
        _load_ref_modules()
        from polar_ref.decoder_utils import active_bit_level, active_llr_level
        from polar_ref.utils import bit_reversed
        self._active_bit_level = active_bit_level
        self._active_llr_level = active_llr_level
        self._bit_reversed = bit_reversed

    def _update_llrs(self, L, B, l):
        for s in range(self.n - self._active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1])

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - self._active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [{
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr

        for phi in range(self.N):
            l = self._bit_reversed(phi, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path["L"], path["B"], l)
                cur_llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    candidates.append({
                        "parent": pidx, "u": 0,
                        "pm": _path_metric_update(path["pm"], cur_llr, 0),
                    })
                else:
                    for u in (0, 1):
                        candidates.append({
                            "parent": pidx, "u": u,
                            "pm": _path_metric_update(path["pm"], cur_llr, u),
                        })

            candidates.sort(key=lambda c: c["pm"])
            candidates = candidates[:self.list_size]

            new_paths = []
            for cand in candidates:
                parent = paths[cand["parent"]]
                new_path = {
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "pm": cand["pm"],
                    "u_hat": parent["u_hat"].copy(),
                }
                new_path["u_hat"][l] = cand["u"]
                new_path["B"][l, self.n] = cand["u"]
                self._update_bits(new_path["B"], l)
                new_paths.append(new_path)

            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
