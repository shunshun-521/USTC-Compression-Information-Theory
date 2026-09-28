"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=np.int64):
        fb = (reg >> (crc_length - 1)) ^ int(b)
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int64)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int64
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（L/C 矩阵 + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_ind = np.asarray(frozen_bits, dtype=np.float64)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length

    def _new_path(self, llr_ch):
        L = np.zeros((self.n + 1, self.N), dtype=np.float64)
        C = np.zeros((self.n, self.N), dtype=np.float64)
        L[self.n, :] = llr_ch
        return {"pm": 0.0, "L": L, "C": C, "u_hat": np.zeros(self.N, dtype=int)}

    def _clone(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "C": path["C"].copy(),
            "u_hat": path["u_hat"].copy(),
        }

    def _update_llr(self, path, phi):
        L, C = path["L"], path["C"]
        layer = 0
        while layer < self.n and (phi >> layer) & 1:
            layer += 1
        for l in range(layer, self.n):
            step = 1 << (self.n - l - 1)
            for i in range(0, self.N, 2 * step):
                La = L[l + 1, i]
                Lb = L[l + 1, i + step]
                L[l, i] = f_operation(La, Lb)
                L[l, i + step] = g_operation(La, Lb, C[l, i])

    def _bit_update(self, path, phi, u_bit):
        C = path["C"]
        C[0, phi] = u_bit
        path["u_hat"][phi] = int(u_bit)
        layer = 0
        while layer < self.n - 1 and (phi >> layer) & 1:
            idx = phi >> (layer + 1)
            C[layer + 1, idx] = C[layer, 2 * idx] ^ C[layer, 2 * idx + 1]
            layer += 1

    def _pm_penalty(self, llr, u):
        if llr == 0:
            u_hard = 1.0
        else:
            u_hard = 0.0 if llr >= 0 else 1.0
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode_recursive(llr_ch, self.frozen_ind)
            return u, 0.0

        llr_ch = -np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            expanded = []
            for path in paths:
                self._update_llr(path, phi)
                llr = path["L"][0, phi]
                if self.frozen_ind[phi] == 1:
                    new_p = self._clone(path)
                    new_p["pm"] += self._pm_penalty(llr, 0)
                    self._bit_update(new_p, phi, 0)
                    expanded.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = self._clone(path)
                        new_p["pm"] += self._pm_penalty(llr, u)
                        self._bit_update(new_p, phi, u)
                        expanded.append(new_p)
            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
