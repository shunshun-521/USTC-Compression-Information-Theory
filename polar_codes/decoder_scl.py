"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _SCDState,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_int,
    _lower_llr,
    _upper_llr,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits(info_bits, poly, crc_len):
    reg = np.zeros(crc_len, dtype=int)
    for bit in info_bits:
        fb = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if fb:
            for j in range(crc_len):
                if (poly >> (crc_len - 1 - j)) & 1:
                    reg[j] ^= fb
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    crc = _crc_bits(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_bits(bits, poly, crc_length)
    return np.all(rem == 0)


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组引用）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen)[0]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u": np.zeros(self.N, dtype=int)}

    def _path_update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + half, s])
                else:
                    top_bit = B[j - half, s + 1]
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - half, s], top_bit)

    def _path_update_bits(self, path, l):
        if l < self.N // 2:
            return
        B = path["B"]
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    B[j - half, s - 1] = int(B[j, s]) ^ int(B[j - half, s])
                    B[j, s - 1] = B[j, s]

    def _penalty(self, llr, bit):
        """与 LLR 不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed_int(i, self.n)
            new_paths = []

            for path in paths:
                self._path_update_llrs(path, l)
                llr_bit = path["L"][l, self.n]
                if np.isnan(llr_bit):
                    llr_bit = 0.0

                if self.frozen[l]:
                    bit = 0
                    pm = path["pm"] + self._penalty(llr_bit, bit)
                    p2 = {
                        "L": path["L"],
                        "B": path["B"].copy(),
                        "pm": pm,
                        "u": path["u"].copy(),
                    }
                    p2["B"][l, self.n] = bit
                    p2["u"][l] = bit
                    self._path_update_bits(p2, l)
                    new_paths.append(p2)
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._penalty(llr_bit, bit)
                        p2 = {
                            "L": path["L"],
                            "B": path["B"].copy(),
                            "pm": pm,
                            "u": path["u"].copy(),
                        }
                        p2["B"][l, self.n] = bit
                        p2["u"][l] = bit
                        self._path_update_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

        best_crc = None
        best_pm = float("inf")
        best_u = paths[0]["u"]

        for path in paths:
            u = path["u"]
            pm = path["pm"]
            if self.crc_length > 0:
                k_info = len(self.info_idx) - self.crc_length
                payload = u[self.info_idx[:k_info]]
                if crc_check(
                    np.concatenate([payload, u[self.info_idx[k_info:]]]),
                    self.crc_length,
                ):
                    if pm < best_pm:
                        best_pm = pm
                        best_u = u
                        best_crc = True
            elif pm < best_pm:
                best_pm = pm
                best_u = u

        if self.crc_length > 0 and best_crc is None:
            best_u = min(paths, key=lambda p: p["pm"])["u"]
            best_pm = min(paths, key=lambda p: p["pm"])["pm"]
        else:
            best_pm = best_pm if best_pm < float("inf") else paths[0]["pm"]

        return best_u.astype(int), best_pm
