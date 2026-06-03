"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _hard_decision,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversed_index


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
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric_penalty(llr, u):
    u_hard = _hard_decision(llr)
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_mask = ~self.frozen_bits

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.float64)
        L[:, 0] = llr_ch
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, use_min_sum=True)
                cur_llr = path["L"][l, self.n]

                if l in self.frozen_set:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + _path_metric_penalty(cur_llr, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    _update_bits(new_path["B"], l, self.n)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + _path_metric_penalty(cur_llr, u),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["u_hat"][l] = u
                        new_path["B"][l, self.n] = u
                        _update_bits(new_path["B"], l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best_crc = None
        if self.crc_length > 0:
            for path in paths:
                info_bits = path["u_hat"][self.info_mask]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path

        best = best_crc if best_crc is not None else min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
