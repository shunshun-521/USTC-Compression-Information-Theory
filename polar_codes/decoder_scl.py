"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly
    return reg


def _crc_poly(crc_length):
    # CRC-8 (0x07) 的位反射多项式表示
    if crc_length == 8:
        return 0xE0
    return 0xA001


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    reg = _crc_remainder(padded, poly, crc_length)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（PSCD 框架，路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, u_val):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u_val == u_from_llr else abs(llr)

    def _crc_passes(self, u_hat):
        if self.crc_length == 0:
            return True
        return crc_check(u_hat[self.info_indices], self.crc_length)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "u_hat": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]

                if l in self.frozen_set:
                    pm = path["pm"] + self._path_metric_penalty(llr, 0)
                    new_path = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, n] = 0
                    _update_bits(new_path["B"], l, n)
                    new_paths.append(new_path)
                else:
                    for u_val in (0, 1):
                        pm = path["pm"] + self._path_metric_penalty(llr, u_val)
                        new_path = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["u_hat"][l] = u_val
                        new_path["B"][l, n] = u_val
                        _update_bits(new_path["B"], l, n)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_paths = [p for p in paths if self._crc_passes(p["u_hat"])]
        best = min(crc_paths if crc_paths else paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
