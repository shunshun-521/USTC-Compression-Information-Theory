"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit) & 1
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "u_hat": np.full(N, -1, dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                L = path["L"].copy()
                B = path["B"].copy()
                _update_llrs(L, B, l, n)
                llr = L[l, n]

                if l in self.frozen_set:
                    pm = path["pm"] + self._path_metric_penalty(llr, 0)
                    B[l, n] = 0
                    _update_bits(B, l, n)
                    u_hat = path["u_hat"].copy()
                    u_hat[l] = 0
                    new_paths.append({"pm": pm, "L": L, "B": B, "u_hat": u_hat})
                else:
                    for bit in (0, 1):
                        Lb = L.copy()
                        Bb = B.copy()
                        pm = path["pm"] + self._path_metric_penalty(llr, bit)
                        Bb[l, n] = bit
                        _update_bits(Bb, l, n)
                        u_hat = path["u_hat"].copy()
                        u_hat[l] = bit
                        new_paths.append({"pm": pm, "L": Lb, "B": Bb, "u_hat": u_hat})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_mask = self.frozen_bits == 0
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][info_mask], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        u_hat = np.zeros(N, dtype=int)
        for i in range(N):
            u_hat[i] = int(best["B"][i, n]) if not np.isnan(best["B"][i, n]) else int(best["u_hat"][i])
        return u_hat, best["pm"]
