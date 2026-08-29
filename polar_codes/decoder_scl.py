"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed_value,
    upper_llr,
    lower_llr,
    hard_decision,
    active_llr_level,
    active_bit_level,
    _update_llrs,
    _update_bits,
    _frozen_indices,
)


CRC8_POLY = np.array([1, 1, 0, 1, 1, 0, 0, 1, 1], dtype=np.int8)
CRC16_POLY = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8
)


def _crc_remainder(bits, poly):
    bits = np.asarray(bits, dtype=np.int8).copy()
    r = len(poly) - 1
    for i in range(len(bits) - r):
        if bits[i] == 1:
            bits[i:i + len(poly)] ^= poly
    return bits[-r:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]), poly
    )
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    data = np.concatenate([bits[:-crc_length], np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_remainder(data, poly)
    return np.array_equal(remainder, bits[-crc_length:])


class PathState:
    """SCL 单条路径状态"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        new = PathState.__new__(PathState)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_indices(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~np.asarray(frozen_bits, dtype=bool))[0]

    def _pm_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [PathState(N, n, llr_ch)]

        for l in [bit_reversed_value(i, n) for i in range(N)]:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    child = path.copy()
                    child.pm = self._pm_update(path.pm, llr, 0)
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    _update_bits(child.B, l, n, N)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm = self._pm_update(path.pm, llr, u)
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        _update_bits(child.B, l, n, N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
