"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    sc_decode,
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = _crc_remainder(payload, poly, crc_length)
    received = 0
    for i in range(crc_length):
        received = (received << 1) | bits[-crc_length + i]
    return expected == received


def _pm_update(pm, llr, u):
  hard = 0 if llr >= 0 else 1
  if u != hard:
      pm += abs(llr)
  return pm


class SCLDecoder:
    """SCL 译码器（置换 SC 结构 + 路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

        if list_size == 1 and crc_length == 0:
            self._use_sc = True
        else:
            self._use_sc = False

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self._use_sc:
            return sc_decode(llr_ch, self.frozen_bits), 0.0
        return self._decode_scl(llr_ch)

    def _decode_scl(self, llr_ch):
        N = self.N
        n = self.n
        Lsize = self.list_size

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=int),
            "u_hat": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for step, l in enumerate(self.decode_order):
            new_paths = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]

                if l in self.frozen_set:
                    pm = _pm_update(path["pm"], llr, 0)
                    child = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["B"][l, n] = 0
                    child["u_hat"][l] = 0
                    _update_bits(child["B"], l, n)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = _pm_update(path["pm"], llr, u)
                        child = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["B"][l, n] = u
                        child["u_hat"][l] = u
                        _update_bits(child["B"], l, n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:Lsize]

        return self._select_best_path(paths)

    def _select_best_path(self, paths):
        if self.crc_length > 0 and self.info_indices is not None:
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
