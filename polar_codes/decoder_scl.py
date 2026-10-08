"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _update_llrs, _update_bits
from decoder_logdomain import bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_step(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    fb = ((reg >> (crc_length - 1)) ^ bit) & 1
    reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    return reg & mask


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=int):
        reg = _crc_step(reg, int(b), poly, crc_length)
    return reg


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    bits = list(np.asarray(info_bits, dtype=int))
    reg = 0
    for b in bits:
        reg = _crc_step(reg, int(b), poly, crc_length)
    for i in range(crc_length - 1, -1, -1):
        bits.append((reg >> i) & 1)
    return np.asarray(bits, dtype=int)


def crc_check(bits, crc_length=8):
    if len(bits) < crc_length:
        return False
    return _crc_remainder(bits, crc_length) == 0


def _branch_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L, B = path["L"], path["B"]
                _update_llrs(L, B, l, n, N)
                llr = L[l, n]

                if self.frozen_bits[l]:
                    pen = _branch_penalty(llr, 0)
                    B[l, n] = 0
                    _update_bits(B, l, n, N)
                    uh = path["u_hat"].copy()
                    uh[l] = 0
                    new_paths.append(
                        {
                            "pm": path["pm"] + pen,
                            "L": L.copy(),
                            "B": B.copy(),
                            "u_hat": uh,
                        }
                    )
                else:
                    for u in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        pen = _branch_penalty(llr, u)
                        Bc[l, n] = u
                        _update_bits(Bc, l, n, N)
                        uh = path["u_hat"].copy()
                        uh[l] = u
                        new_paths.append(
                            {
                                "pm": path["pm"] + pen,
                                "L": Lc,
                                "B": Bc,
                                "u_hat": uh,
                            }
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            u_hat = path["u_hat"]
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc[1]:
                        best_crc = (u_hat, path["pm"])
            if best_pm is None or path["pm"] < best_pm[1]:
                best_pm = (u_hat, path["pm"])

        chosen = best_crc if best_crc is not None else best_pm
        return chosen[0].astype(int), chosen[1]
