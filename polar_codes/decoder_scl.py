"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    clip_llr,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length <= 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError("N must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr(self, path, l):
        return path.L[l, self.n]

    def _branch_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)。
        """
        llr_ch = clip_llr(np.asarray(llr_ch, dtype=np.float64))
        br = np.array([_bit_reversed(i, self.n) for i in range(self.N)], dtype=int)
        llr_ch = llr_ch[br]

        paths = [_Path(self.N, self.n, llr_ch.copy())]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = self._path_llr(path, l)

                if l in self.frozen_set:
                    pen = self._branch_penalty(llr, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = _Path(self.N, self.n, path.L[:, 0].copy())
                        new_path.L[:, 1:] = path.L[:, 1:].copy()
                        new_path.B[:, 1:] = path.B[:, 1:].copy()
                        new_path.B[l, self.n] = bit
                        new_path.pm = path.pm + self._branch_penalty(llr, bit)
                        _update_bits(new_path.B, l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_paths = []
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    best_paths.append(path)
            else:
                best_paths.append(path)

        if not best_paths:
            best_paths = paths

        best = min(best_paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
