"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reverse,
    _lower_llr,
    _update_bits,
    _update_llrs,
)


def _crc_mod(bits, poly, crc_len):
    reg = 0
    mask = (1 << crc_len) - 1
    for b in bits:
        fb = (reg >> (crc_len - 1)) & 1
        reg = ((reg << 1) | int(b)) & mask
        if fb:
            reg ^= poly
    for _ in range(crc_len):
        fb = (reg >> (crc_len - 1)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_mod(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_mod(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径级 Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=int)}

    def _path_penalty(self, llr_val, bit):
        if llr_val >= 0:
            return 0.0 if bit == 0 else abs(llr_val)
        return 0.0 if bit == 1 else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reverse(i, self.n)
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                llr_bit = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    pm = path["pm"] + self._path_penalty(llr_bit, 0)
                    candidates.append((pm, path, 0))
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._path_penalty(llr_bit, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, base, bit in selected:
                child = {
                    "pm": pm,
                    "L": base["L"].copy(),
                    "B": base["B"].copy(),
                    "u_hat": base["u_hat"].copy(),
                }
                if self.frozen_bits[l]:
                    child["B"][l, self.n] = 0
                else:
                    child["B"][l, self.n] = bit
                child["u_hat"][l] = child["B"][l, self.n]
                _update_bits(child["B"], l, self.n, self.N)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [p for p in paths if crc_check(p["u_hat"][info_positions], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
