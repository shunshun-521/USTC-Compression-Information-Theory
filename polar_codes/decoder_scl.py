"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed,
    _active_llr_level, _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length=8):
    """计算 CRC 余数（MSB-first LFSR）"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    crc = 0
    for bit in bits:
        msb = (crc >> (crc_length - 1)) & 1
        crc = ((crc << 1) & mask) ^ (poly if (msb ^ int(bit)) else 0)
    return crc


def crc_encode(info_bits, crc_length=8):
    """CRC 编码（MSB-first LFSR）"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """CRC 校验"""
    if crc_length == 0:
        return True
    return _crc_remainder(bits, crc_length) == 0


class Path:
    """SCL 单条路径"""

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch.astype(np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            bs = 2 ** (s + 1)
            br = bs // 2
            for j in range(l, self.N, bs):
                if j % bs < br:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + br, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - br, s], path.L[j, s], path.B[j - br, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            bs = 2 ** s
            br = bs // 2
            for j in range(l, -1, -bs):
                if j % bs >= br:
                    path.B[j - br, s - 1] = int(path.B[j, s]) ^ int(path.B[j - br, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _copy_path(self, src):
        dst = Path(self.N, self.n, np.zeros(self.N))
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    new_path.pm += penalty
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    new_paths.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._copy_path(path)
                        if u_bit == 0:
                            penalty = abs(llr_val) if llr_val < 0 else 0.0
                        else:
                            penalty = abs(llr_val) if llr_val >= 0 else 0.0
                        new_path.pm += penalty
                        new_path.B[l, self.n] = u_bit
                        new_path.u_hat[l] = u_bit
                        self._update_bits(new_path, l)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            crc_paths = [
                p for p in paths
                if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            best = min(crc_paths if crc_paths else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
