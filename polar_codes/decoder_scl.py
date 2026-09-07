"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
    sc_decode,
)
from encoder import bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = _crc_poly(crc_length)
    reg = 0
    msg = np.concatenate(
        [np.asarray(info_bits, dtype=np.int8), np.zeros(crc_length, dtype=np.int8)]
    )
    for bit in msg:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([np.asarray(info_bits, dtype=np.int8), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in np.asarray(bits, dtype=np.int8):
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [
            {
                "pm": 0.0,
                "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
                "u_hat": np.zeros(self.N, dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr_leaf = path["L"][l, self.n]

                if l in self.frozen_set:
                    hard = 0 if llr_leaf >= 0 else 1
                    penalty = 0.0 if hard == 0 else abs(llr_leaf)
                    new_path = {
                        "pm": path["pm"] + penalty,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][l, self.n] = 0
                    new_path["u_hat"][l] = 0
                    _update_bits(new_path["B"], l, self.n)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        hard = 0 if llr_leaf >= 0 else 1
                        penalty = 0.0 if u_bit == hard else abs(llr_leaf)
                        new_path = {
                            "pm": path["pm"] + penalty,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["B"][l, self.n] = u_bit
                        new_path["u_hat"][l] = u_bit
                        _update_bits(new_path["B"], l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid or paths, key=lambda x: x["pm"])
        else:
            best = min(paths, key=lambda x: x["pm"])

        return best["u_hat"], best["pm"]
