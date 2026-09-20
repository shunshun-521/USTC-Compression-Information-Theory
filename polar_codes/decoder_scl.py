"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import _active_bit_level, _active_llr_level, _update_bits, _update_llrs
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n, N)
                llr = L[l, n]
                if l in self.frozen_set:
                    pm = _pm_update(path["pm"], llr, 0)
                    np_ = copy.deepcopy(path)
                    np_["pm"] = pm
                    np_["B"][l, n] = 0
                    _update_bits(np_["B"], l, n, N)
                    new_paths.append(np_)
                else:
                    for u in (0, 1):
                        pm = _pm_update(path["pm"], llr, u)
                        np_ = copy.deepcopy(path)
                        np_["pm"] = pm
                        np_["B"][l, n] = u
                        _update_bits(np_["B"], l, n, N)
                        new_paths.append(np_)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p["B"][:, n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        u_hat = best["B"][:, n].astype(int)
        return u_hat, best["pm"]
