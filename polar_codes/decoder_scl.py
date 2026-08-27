"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Vangala 置换 SC
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    _frozen_set,
    _update_llrs,
    _update_bits,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = _frozen_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen))

    @staticmethod
    def _penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "u_hat": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n, N)
                llr_val = path["L"][l, n]

                if l in self.frozen:
                    pm = path["pm"] + self._penalty(llr_val, 0)
                    child = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["B"][l, n] = 0
                    child["u_hat"][l] = 0
                    _update_bits(child["B"], l, n, N)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + self._penalty(llr_val, u_bit)
                        child = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["B"][l, n] = u_bit
                        child["u_hat"][l] = u_bit
                        _update_bits(child["B"], l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                return best["u_hat"], best["pm"]

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
