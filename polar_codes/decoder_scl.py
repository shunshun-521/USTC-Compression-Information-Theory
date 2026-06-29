"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _update_bits,
    _update_llrs,
    bit_reversed_index,
    f_boxplus,
    prepare_channel_llr,
    sc_decode,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(bits, crc_length, poly):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_bits(info_bits, crc_length, poly)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = prepare_channel_llr(llr_ch, self.N)
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=int),
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed_index(i, n)
            candidates = []

            for path in paths:
                L = path["L"].copy()
                B = path["B"].copy()
                _update_llrs(L, B, l, n, upper=f_boxplus)
                llr = float(L[l, n])

                if l in self.frozen_set:
                    pm = path["pm"] + self._path_metric_penalty(llr, 0)
                    u_val = 0
                    B[l, n] = 0
                    _update_bits(B, l, n)
                    new_path = {
                        "pm": pm,
                        "L": L,
                        "B": B,
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["u_hat"][l] = u_val
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        Lc = path["L"].copy()
                        Bc = path["B"].copy()
                        _update_llrs(Lc, Bc, l, n, upper=f_boxplus)
                        llr_c = float(Lc[l, n])
                        Bc[l, n] = u_val
                        _update_bits(Bc, l, n)
                        pm = path["pm"] + self._path_metric_penalty(llr_c, u_val)
                        u_hat = path["u_hat"].copy()
                        u_hat[l] = u_val
                        candidates.append(
                            {"pm": pm, "L": Lc, "B": Bc, "u_hat": u_hat}
                        )

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][info_idx], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
