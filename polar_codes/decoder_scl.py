"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _update_bits,
    _update_llrs,
    prepare_channel_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg & mask


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
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr = prepare_channel_llr(llr_ch)
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=int),
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr

        for phi in range(N):
            l = _bit_reversed_index(phi, n)
            new_paths = []

            for path in paths:
                L = path["L"].copy()
                B = path["B"].copy()
                _update_llrs(L, B, l, n)
                llr0 = L[l, n]

                if l in self.frozen_set:
                    pm = path["pm"] + self._path_metric_penalty(llr0, 0)
                    B[l, n] = 0
                    u_hat = path["u_hat"].copy()
                    u_hat[l] = 0
                    _update_bits(B, l, n)
                    new_paths.append(
                        {"pm": pm, "L": L, "B": B, "u_hat": u_hat}
                    )
                else:
                    for bit in (0, 1):
                        Lb = L.copy()
                        Bb = B.copy()
                        Bb[l, n] = bit
                        u_hat = path["u_hat"].copy()
                        u_hat[l] = bit
                        pm = path["pm"] + self._path_metric_penalty(llr0, bit)
                        _update_bits(Bb, l, n)
                        new_paths.append(
                            {"pm": pm, "L": Lb, "B": Bb, "u_hat": u_hat}
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
