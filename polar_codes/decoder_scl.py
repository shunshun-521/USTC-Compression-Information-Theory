"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
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


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]       # x^8 + x^2 + x + 1
CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # CRC-16-IBM


def _crc_divide(msg, poly):
    msg = list(msg)
    degree = len(poly) - 1
    for i in range(len(msg) - degree):
        if msg[i]:
            for j in range(len(poly)):
                msg[i + j] ^= poly[j]
    return msg[-degree:]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = np.array(_crc_divide(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly), dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_divide(bits, poly)
    return np.all(np.array(remainder) == 0)


class Path:
    """单条 SCL 路径（Lazy Copy）。"""

    __slots__ = ('pm', 'L', 'B', 'active')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.active = True

    def copy(self):
        new = Path(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0])
        new.pm = self.pm
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.active = self.active
        return new


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 u_hat（最优路径）和 pm（路径度量）。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                _update_llrs(l, self.n, path.L, path.B)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pm = path.pm + self._path_metric_penalty(llr, 0)
                    new_path = path.copy()
                    new_path.pm = pm
                    new_path.B[l, self.n] = 0
                    _update_bits(l, self.n, new_path.B)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._path_metric_penalty(llr, bit)
                        new_path = path.copy()
                        new_path.pm = pm
                        new_path.B[l, self.n] = bit
                        _update_bits(l, self.n, new_path.B)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_all = min(paths, key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in sorted(paths, key=lambda p: p.pm):
                u_hat = path.B[:, self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    best_crc = path
                    break

        best = best_crc if best_crc is not None else best_all
        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm
