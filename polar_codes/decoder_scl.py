"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _compute_llr


def _crc_bits(info_bits, crc_length, poly):
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    out = np.zeros(crc_length, dtype=int)
    for i in range(crc_length):
        out[crc_length - 1 - i] = (reg >> i) & 1
    return out


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    crc = _crc_bits(info_bits, crc_length, poly)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    expected = _crc_bits(bits[:-crc_length], crc_length, poly)
    return np.array_equal(expected, bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（惰性 LLR + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path_state(self, llr_ch):
        llrs = np.full((self.n + 1, self.N), -np.inf, dtype=np.float64)
        llrs[self.n, :] = llr_ch
        s = np.full((self.n + 1, self.N), -1, dtype=np.int8)
        return {"llrs": llrs, "s": s, "u_hat": np.zeros(self.N, dtype=int), "pm": 0.0}

    def _path_llr(self, path, phi):
        llrs, s = path["llrs"], path["s"]
        if llrs[0, phi] == -np.inf:
            llrs[0, phi] = _compute_llr(0, phi, llrs, s)
        return llrs[0, phi]

    def _clone_path(self, path):
        return {
            "llrs": path["llrs"].copy(),
            "s": path["s"].copy(),
            "u_hat": path["u_hat"].copy(),
            "pm": path["pm"],
        }

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size
        paths = [self._new_path_state(llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_val = self._path_llr(path, phi)
                if self.frozen_bits[phi]:
                    p2 = self._clone_path(path)
                    p2["u_hat"][phi] = 0
                    p2["s"][0, phi] = 0
                    p2["llrs"][0, phi] = np.inf
                    p2["pm"] = path["pm"] + (0.0 if llr_val >= 0 else abs(llr_val))
                    candidates.append(p2)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0) else abs(llr_val)
                        p2 = self._clone_path(path)
                        p2["u_hat"][phi] = bit
                        p2["s"][0, phi] = bit
                        p2["llrs"][0, phi] = llr_val
                        p2["pm"] = path["pm"] + penalty
                        candidates.append(p2)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:L]

        return self._select_best(paths)

    def _select_best(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                return best["u_hat"], best["pm"]

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
