"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
)


def _crc_bits(bits, poly, width):
    reg = 0
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    remainder = _crc_bits(info_bits, poly, crc_length)
    crc_part = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_part])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    return _crc_bits(bits, poly, crc_length) == 0


def _path_metric_update(pm, llr, bit):
    preferred = 0 if llr >= 0 else 1
    penalty = 0.0 if bit == preferred else abs(llr)
    return pm + penalty


class Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _continue_path(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        path.u_hat[l] = bit
        if l >= self.N / 2:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                for j in range(l, -1, -block_size):
                    if j % block_size >= block_size // 2:
                        path.B[j - block_size // 2, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - block_size // 2, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, path):
        new_path = Path(self.N, self.n, path.L[:, 0])
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch.copy())]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._continue_path(path, l)
                llr = path.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if l in self.frozen_set:
                    new_path = self._clone_path(path)
                    new_path.pm = _path_metric_update(path.pm, llr, 0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm = _path_metric_update(path.pm, llr, bit)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
