"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        crc ^= int(b) << (crc_length - 1)
        if crc & (1 << (crc_length - 1)):
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    crc_bits = np.array(
        [(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC，对全部比特（含 CRC）计算余数应为 0"""
    if crc_length == 0:
        return True
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        crc ^= int(b) << (crc_length - 1)
        if crc & (1 << (crc_length - 1)):
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    return crc == 0


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class _Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时深拷贝 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        N, n = self.N, self.n
        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    pm = path.pm + _pm_penalty(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for bit in (0, 1):
                        pm = path.pm + _pm_penalty(llr, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in survivors:
                child = _Path(N, n)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.u_hat[l] = bit
                child.B[l, n] = bit
                _update_bits(child.B, l, n, N)
                new_paths.append(child)

            paths = new_paths

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            crc_ok = [
                p for p in paths
                if crc_check(p.u_hat[info_mask], self.crc_length)
            ]
            if crc_ok:
                best = min(crc_ok, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
