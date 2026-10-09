"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _bit_reversed,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_lfsr(bits, crc_length):
    """CRC LFSR（多项式 0x07 / 0x8005，MSB 先行）"""
    poly = _crc_poly(crc_length)
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for b in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_lfsr(np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]), crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def _current_llr(self, path, l):
        _update_llrs(path["L"], path["B"], l, self.n, self.N)
        return path["L"][l, self.n]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                llr = self._current_llr(path, l)
                if l in self.frozen_idx:
                    path["pm"] += 0.0 if llr >= 0 else abs(llr)
                    path["u_hat"][l] = 0
                    path["B"][l, self.n] = 0
                    _update_bits(path["B"], l, self.n, self.N)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"],
                            "u_hat": path["u_hat"].copy(),
                        }
                        consistent = (bit == 0 and llr >= 0) or (bit == 1 and llr < 0)
                        child["pm"] += 0.0 if consistent else abs(llr)
                        child["u_hat"][l] = bit
                        child["B"][l, self.n] = bit
                        _update_bits(child["B"], l, self.n, self.N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            if self.crc_length > 0:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path
            if best_pm is None or path["pm"] < best_pm["pm"]:
                best_pm = path

        chosen = best_crc if best_crc is not None else best_pm
        return chosen["u_hat"].copy(), float(chosen["pm"])
