"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from encoder import prepare_channel_llr
from decoder_sc import _f_min_sum


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（每条路径独立 L/R 消息，Lazy Copy 在路径分裂时复制 u 与 R）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self._large = 1e6
        self._alpha = 1.0

    def _left_sweep(self, L, R, u_hat, phi):
        for j in range(self.n, 0, -1):
            s = 1 << (j - 1)
            for block in range(0, self.N, 2 * s):
                for i in range(block, block + s):
                    ip = i + s
                    L[i, j - 1] = _f_min_sum(R[i, j] + L[ip, j], L[i, j], self._alpha)
                    L[ip, j - 1] = _f_min_sum(R[i, j], L[i, j], self._alpha) + L[ip, j]
        return L[phi, 0] + R[phi, 0]

    def decode(self, llr_ch):
        llr_br = prepare_channel_llr(llr_ch)
        N, n = self.N, self.n

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        R0 = np.zeros((N, n + 1), dtype=np.float64)
        L0[:, n] = llr_br
        paths.append({"pm": 0.0, "u": np.zeros(N, dtype=int), "L": L0, "R": R0})

        for phi in range(N):
            candidates = []
            for path in paths:
                L, R, u = path["L"], path["R"], path["u"]
                L[:, n] = llr_br
                R.fill(0.0)
                R[self.frozen_bits, 0] = self._large
                for i in range(phi):
                    if not self.frozen_bits[i]:
                        R[i, 0] = self._large if u[i] == 0 else -self._large
                llr_phi = self._left_sweep(L, R, u, phi)

                if self.frozen_bits[phi]:
                    pm = path["pm"] + (abs(llr_phi) if llr_phi < 0 else 0.0)
                    nu = {
                        "pm": pm,
                        "u": u.copy(),
                        "L": L.copy(),
                        "R": R.copy(),
                    }
                    nu["u"][phi] = 0
                    candidates.append(nu)
                else:
                    for bit in (0, 1):
                        pm = path["pm"]
                        hard = 0 if llr_phi >= 0 else 1
                        if bit != hard:
                            pm += abs(llr_phi)
                        nu = {
                            "pm": pm,
                            "u": u.copy(),
                            "L": L.copy(),
                            "R": R.copy(),
                        }
                        nu["u"][phi] = bit
                        candidates.append(nu)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        info_positions = np.where(~self.frozen_bits)[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][info_positions], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"].astype(int), best["pm"]
