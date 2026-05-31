"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_boxplus,
    g_operation,
    _channel_llr_to_decoder,
    sc_decode,
    LLR_MAX,
)
from utils import crc_encode as utils_crc_encode
from utils import crc_check as utils_crc_check


def crc_encode(info_bits, crc_length=8):
    return utils_crc_encode(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    return utils_crc_check(bits, crc_length)


def _sc_update_llr(L, B, x, n, N):
    for j in range(n - 1, -1, -1):
        s = 2 ** (n - j)
        t = s // 2
        for i in range(x, N, s):
            if t > (i % s):
                L[i, j] = f_boxplus(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _sc_update_bits(B, x, n):
    for j in range(n):
        s = 2 ** (n - j)
        t = s // 2
        if (x % s) >= t:
            i = x - t
            B[i, j + 1] = B[i, j] ^ B[i + t, j]


def _llr_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器：每路径维护完整 L/B 状态（与 SC 矩阵算法一致）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, self.n] = llr_ch
        return {"pm": 0.0, "u": np.zeros(self.N, dtype=int), "L": L, "B": B}

    def decode(self, llr_ch):
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_ch = np.clip(_channel_llr_to_decoder(llr_ch), -LLR_MAX, LLR_MAX)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                _sc_update_llr(path["L"], path["B"], phi, self.n, self.N)
                llr_phi = path["L"][phi, 0]

                if self.frozen_bits[phi]:
                    pen = _llr_penalty(llr_phi, 0)
                    path["pm"] += pen
                    path["u"][phi] = 0
                    path["B"][phi, 0] = 0
                    _sc_update_bits(path["B"], phi, self.n)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        p = {
                            "pm": path["pm"] + _llr_penalty(llr_phi, bit),
                            "u": path["u"].copy(),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        p["u"][phi] = bit
                        p["B"][phi, 0] = bit
                        _sc_update_bits(p["B"], phi, self.n)
                        candidates.append(p)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = self._select_best(paths)
        return best["u"].copy(), best["pm"]

    def _select_best(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                if crc_check(p["u"][self.info_indices], self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda x: x["pm"])
        return min(paths, key=lambda x: x["pm"])
