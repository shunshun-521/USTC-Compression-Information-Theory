"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _INF,
    _b_check,
    _compute_llr,
    _s_updater,
    f_operation,
    g_operation,
)


_CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = _CRC_POLY[crc_length]
    reg = 0
    for b in bits:
        reg = (reg << 1) | int(b)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_penalty(llr, u):
    """路径度量惩罚：判决与 LLR 符号不一致时加 |LLR|"""
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, llr_ch):
        llrs = np.full((self.n + 1, self.N), -_INF, dtype=np.float64)
        llrs[self.n, :] = llr_ch
        s = np.full((self.n + 1, self.N), -1, dtype=np.int8)
        return {"llrs": llrs, "s": s, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def _path_llr(self, path, phi):
        return _compute_llr(0, phi, path["llrs"], path["s"])

    def _path_decide(self, path, phi, u):
        path["u_hat"][phi] = u
        path["s"][0, phi] = u

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = self._path_llr(path, phi)
                if self.frozen_bits[phi]:
                    new_path = {
                        "llrs": path["llrs"].copy(),
                        "s": path["s"].copy(),
                        "pm": path["pm"] + _pm_penalty(llr, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    self._path_decide(new_path, phi, 0)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = {
                            "llrs": path["llrs"].copy(),
                            "s": path["s"].copy(),
                            "pm": path["pm"] + _pm_penalty(llr, u),
                            "u_hat": path["u_hat"].copy(),
                        }
                        self._path_decide(new_path, phi, u)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [p for p in paths if self._crc_pass(p)]
            best = min(crc_pass or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]

    def _crc_pass(self, path):
        info_bits = path["u_hat"][~self.frozen_bits]
        return crc_check(info_bits, self.crc_length)
