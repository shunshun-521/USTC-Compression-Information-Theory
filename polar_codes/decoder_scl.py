"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc8_run(bits, init=0xFF):
    """CRC-8 (poly 0x07) MSB-first，init=0xFF"""
    crc = init
    for b in bits:
        fb = ((crc >> 7) ^ int(b)) & 1
        crc = ((crc << 1) & 0xFF) ^ (_CRC8_POLY if fb else 0)
    return crc


def _crc16_run(bits, init=0xFFFF):
    """CRC-16 (poly 0x8005) MSB-first"""
    crc = init
    for b in bits:
        fb = ((crc >> 15) ^ int(b)) & 1
        crc = ((crc << 1) & 0xFFFF) ^ (_CRC16_POLY if fb else 0)
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        rem = _crc8_run(info_bits)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        rem = _crc16_run(info_bits)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        return _crc8_run(bits) == 0
    return _crc16_run(bits) == 0


class SCLDecoder:
    """SCL 译码器（路径度量 + Lazy Copy 浅拷贝）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = (
            precompute_sc_indices(N)
        )

    @staticmethod
    def _penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        inv_br = np.argsort(
            np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)
        )
        llr_ch = llr_ch[inv_br]

        paths = [
            {
                "pm": 0.0,
                "u": np.zeros(N, dtype=int),
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, n, N)
                llr0 = path["L"][l, n]
                if self.frozen_bits[l]:
                    for bit in (0,):
                        p = self._fork_path(path, l, bit, llr0)
                        new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = self._fork_path(path, l, bit, llr0)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

            for path in paths:
                _update_bits(path["B"], l, n, N)

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            candidates = valid if valid else paths
        else:
            candidates = paths

        best = min(candidates, key=lambda p: p["pm"])
        return best["u"], best["pm"]

    def _fork_path(self, path, l, bit, llr0):
        p = {
            "pm": path["pm"] + self._penalty(llr0, bit),
            "u": path["u"].copy(),
            "L": path["L"].copy(),
            "B": path["B"].copy(),
        }
        p["B"][l, self.n] = bit
        p["u"][l] = bit
        return p
