"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_compute(bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for b in np.asarray(bits, dtype=int).ravel():
        fb = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = ((reg << 1) | int(b)) & mask
        if fb:
            reg ^= poly if crc_length == 8 else (poly >> (16 - crc_length)) & mask
    for _ in range(crc_length):
        fb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly if crc_length == 8 else (poly >> (16 - crc_length)) & mask
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    return np.concatenate([info_bits, _crc_compute(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan, dtype=np.float64),
            "u": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        def update_llrs(path, l):
            Lmat, Bmat = path["L"], path["B"]
            for s in range(n - active_llr_level(l, n), n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        Lmat[j, s + 1] = f_operation(Lmat[j, s], Lmat[j + branch_size, s])
                    else:
                        Lmat[j, s + 1] = g_operation(
                            Lmat[j - branch_size, s], Lmat[j, s], Bmat[j - branch_size, s + 1]
                        )

        def update_bits(path, l):
            Bmat = path["B"]
            if l < N // 2:
                return
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        Bmat[j - branch_size, s - 1] = int(Bmat[j, s]) ^ int(Bmat[j - branch_size, s])
                        Bmat[j, s - 1] = Bmat[j, s]

        for i in range(N):
            l = bit_reversed_index(i, n)
            candidates = []

            for pidx, path in enumerate(paths):
                update_llrs(path, l)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    pen = self._path_penalty(llr, 0)
                    path["pm"] += pen
                    path["u"][l] = 0
                    path["B"][l, n] = 0
                    update_bits(path, l)
                else:
                    for bit in (0, 1):
                        candidates.append((path["pm"] + self._path_penalty(llr, bit), pidx, bit))

            if not self.frozen_bits[l]:
                candidates.sort(key=lambda x: x[0])
                new_paths = []
                for pm, pidx, bit in candidates[:L]:
                    if len(new_paths) < L:
                        if len([c for c in candidates[:L] if c[1] == pidx]) == 1 and any(
                            id(p) == id(paths[pidx]) for p in new_paths
                        ):
                            np_ = paths[pidx]
                        else:
                            src = paths[pidx]
                            np_ = {
                                "pm": pm,
                                "L": src["L"].copy(),
                                "B": src["B"].copy(),
                                "u": src["u"].copy(),
                            }
                        np_["pm"] = pm
                        np_["u"][l] = bit
                        np_["B"][l, n] = bit
                        update_bits(np_, l)
                        new_paths.append(np_)
                paths = new_paths if new_paths else paths

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best["u"].copy(), best["pm"]
