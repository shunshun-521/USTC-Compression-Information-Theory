"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation

CRC8_POLY = [1, 1, 1, 0, 0, 0, 0, 0, 1]          # x^8 + x^2 + x + 1
CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # CRC-16-IBM


def _crc_mod2(bits, poly):
    n = len(poly)
    reg = list(bits) + [0] * (n - 1)
    for i in range(len(bits)):
        if reg[i]:
            for j in range(n):
                reg[i + j] ^= poly[j]
    return reg[len(bits): len(bits) + n - 1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_mod2(info_bits.tolist(), poly)
    return np.concatenate([info_bits, np.array(remainder, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return all(x == 0 for x in _crc_mod2(bits.tolist(), poly))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, u):
        penalty = 0.0 if (llr >= 0 and u == 0) or (llr < 0 and u == 1) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self.br]
        n, N = self.n, self.N

        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n, N)
                llr = L[l, n]

                if l in self.frozen_set:
                    cand = [(0, self._pm_update(path["pm"], llr, 0))]
                else:
                    cand = [(0, self._pm_update(path["pm"], llr, 0)),
                            (1, self._pm_update(path["pm"], llr, 1))]

                for u, pm in cand:
                    np_path = {
                        "L": L.copy(),
                        "B": copy.deepcopy(B),
                        "pm": pm,
                    }
                    np_path["B"][l, n] = u
                    _update_bits(np_path["B"], l, n, N)
                    new_paths.append(np_path)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p["B"][:, n].astype(np.int8)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append(p)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["B"][:, n].astype(np.int8), best["pm"]
