"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _update_bits,
    _update_llrs,
    bit_reversed_index,
    sc_decode,
    sc_path_metric,
)
from utils import crc_encode_bits, crc_check_bits


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = crc_encode_bits(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验"""
    return crc_check_bits(np.asarray(bits, dtype=int), crc_length)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 路径管理）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_perm = llr_ch[br]
        paths = [self._init_path(llr_perm)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                llr_val = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path["pm"] += sc_path_metric(llr_val, 0)
                    new_path["B"][l, self.n] = 0
                    _update_bits(new_path["B"], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path["pm"] += sc_path_metric(llr_val, u_bit)
                        new_path["B"][l, self.n] = u_bit
                        _update_bits(new_path["B"], l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = self._select_best_path(paths)
        return best["B"][:, self.n].astype(int), best["pm"]

    def _init_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B}

    def _copy_path(self, path):
        return {"pm": path["pm"], "L": path["L"].copy(), "B": path["B"].copy()}

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path["B"][:, self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                return min(valid, key=lambda p: p["pm"])
        return min(paths, key=lambda p: p["pm"])
