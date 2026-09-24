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
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1], dtype=int)
CRC16_POLY = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int)


def _crc_mod2(data_bits, gen):
    data = list(map(int, data_bits))
    n = len(gen) - 1
    for i in range(len(data) - n):
        if data[i] == 1:
            for j in range(len(gen)):
                if i + j < len(data):
                    data[i + j] ^= gen[j]
    return np.array(data[-n:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。CRC-8: 0x07; CRC-16: 0x8005"""
    info_bits = np.asarray(info_bits, dtype=int)
    gen = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_mod2(padded, gen)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    gen = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return np.all(_crc_mod2(bits, gen) == 0)


def _pm_update(pm, llr, u):
    expected = 0 if llr >= 0 else 1
    if u != expected:
        pm += abs(llr)
    return pm


class _Path:
    """SCL 路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = _Path.__new__(_Path)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        N = self.N
        n = self.n

        active = [_Path(N, n, llr)]

        for l in [_bit_reversed(i, n) for i in range(N)]:
            candidates = []
            for path in active:
                _update_llrs(path.L, path.B, l, n)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    u = 0
                    path.pm = _pm_update(path.pm, llr, u)
                    path.u_hat[l] = u
                    path.B[l, n] = u
                    _update_bits(path.B, l, n)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm = _pm_update(path.pm, llr, u)
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        _update_bits(child.B, l, n)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            active = candidates[: self.list_size]

        if self.crc_length > 0:
            info_pos = np.where(self.frozen_bits == 0)[0]
            valid = [
                p for p in active
                if crc_check(p.u_hat[info_pos], self.crc_length)
            ]
            best = min(valid if valid else active, key=lambda p: p.pm)
        else:
            best = min(active, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
