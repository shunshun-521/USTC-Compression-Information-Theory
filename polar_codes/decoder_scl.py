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
    _inverse_bit_reversal,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]
_CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1]


def _gf2_poly_div(dividend, divisor):
    dividend = list(dividend)
    divisor = list(divisor)
    n = len(divisor)
    m = len(dividend)
    for i in range(m - n + 1):
        if dividend[i] == 1:
            for j in range(n):
                dividend[i + j] ^= divisor[j]
    return dividend[m - n + 1:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    msg = list(info_bits) + [0] * crc_length
    remainder = _gf2_poly_div(msg, gen)
    return np.array(list(info_bits) + remainder, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    remainder = _gf2_poly_div(list(bits), gen)
    return all(x == 0 for x in remainder)


def _path_metric_penalty(llr, u):
    """路径度量惩罚。"""
    u_preferred = 0 if llr >= 0 else 1
    return 0.0 if u == u_preferred else abs(llr)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.inv_brp = _inverse_bit_reversal(N)
        self.brp = bit_reversal_permutation(N)

    def _compute_llr_at_phi(self, path, phi):
        """计算当前比特 phi 的 LLR。"""
        l = _bit_reversed_index(phi, self.n)
        L = path["L"]
        B = path["B"]
        _update_llrs(L, B, l, self.n)
        return L[l, self.n]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            from decoder_sc import sc_decode
            u_hat, pm = sc_decode(llr_ch, self.frozen_bits), 0.0
            return u_hat, pm

        paths = [{
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int32),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            src_idx = self.inv_brp[l]
            new_paths = []

            for path in paths:
                llr = self._compute_llr_at_phi(path, phi)

                if self.frozen_bits[src_idx]:
                    new_p = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + _path_metric_penalty(llr, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_p["B"][l, self.n] = 0
                    _update_bits(new_p["B"], l, self.n)
                    new_p["u_hat"][src_idx] = 0
                    new_paths.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + _path_metric_penalty(llr, u),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_p["B"][l, self.n] = u
                        _update_bits(new_p["B"], l, self.n)
                        new_p["u_hat"][src_idx] = u
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][info_mask], self.crc_length)
            ]
            best = min(valid, key=lambda p: p["pm"]) if valid else paths[0]
        else:
            best = paths[0]

        return best["u_hat"], best["pm"]
