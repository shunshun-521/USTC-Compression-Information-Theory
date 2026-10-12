"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reverse_int,
    _update_llrs,
    _update_bits,
    f_operation,
    g_operation,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(b)) & 1
        reg = ((reg << 1) & mask)
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_ch_idx = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [
            {
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, n] = llr_ch

        for i in range(N):
            l = _bit_reverse_int(i, n)
            new_paths = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, n, N)
                llr_bit = path["L"][l, 0]
                if self.frozen_bits[l]:
                    pm = self._pm_update(path["pm"], llr_bit, 0)
                    npth = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": pm,
                    }
                    npth["B"][l, 0] = 0
                    _update_bits(npth["B"], l, n, N)
                    new_paths.append(npth)
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path["pm"], llr_bit, u)
                        npth = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": pm,
                        }
                        npth["B"][l, 0] = u
                        _update_bits(npth["B"], l, n, N)
                        new_paths.append(npth)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            ok = []
            for p in paths:
                u_hat = p["B"][:, 0]
                if crc_check(u_hat[self.info_ch_idx], self.crc_length):
                    ok.append(p)
            if ok:
                best = min(ok, key=lambda p: p["pm"])

        u_hat = best["B"][:, 0]
        return u_hat.astype(int), best["pm"]
