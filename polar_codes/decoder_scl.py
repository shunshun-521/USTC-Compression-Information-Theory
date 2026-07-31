"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _bit_reversed, _update_bits, _update_llrs


CRC8_POLY = 0x107
CRC16_POLY = 0x11005


def _crc_remainder(bits, poly, crc_length):
    msg = 0
    for bit in bits:
        msg = (msg << 1) | int(bit)
    nbits = len(bits)
    for i in range(nbits - 1, crc_length - 1, -1):
        if (msg >> i) & 1:
            msg ^= poly << (i - crc_length)
    return msg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly, crc_length
    )
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _current_llr(self, path, l):
        _update_llrs(path.L, path.B, l, self.n)
        return path.L[l, self.n]

    def _path_metric_penalty(self, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for path in paths:
                llr = self._current_llr(path, l)
                if l in self.frozen_set:
                    new_path = _Path(self.N, self.n, llr_ch)
                    new_path.pm = path.pm + self._path_metric_penalty(llr, 0)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat = path.u_hat.copy()
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = _Path(self.N, self.n, llr_ch)
                        new_path.pm = path.pm + self._path_metric_penalty(llr, u_bit)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[l] = u_bit
                        new_path.B[l, self.n] = u_bit
                        _update_bits(new_path.B, l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if not crc_check(info_bits, self.crc_length):
                    continue
            if best_pm is None or path.pm < best_pm:
                best_pm = path.pm
                best_crc = path

        if best_crc is not None:
            return best_crc.u_hat.copy(), best_crc.pm

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
