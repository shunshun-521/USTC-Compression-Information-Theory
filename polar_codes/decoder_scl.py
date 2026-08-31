"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    _bit_reversed,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly_divide(bits, crc_length=8):
    """CRC 多项式除法，返回余数"""
    poly = (1 << crc_length) | (CRC8_POLY if crc_length == 8 else CRC16_POLY)
    msg = list(np.asarray(bits, dtype=int))
    n = len(msg) - crc_length
    for i in range(n):
        if msg[i] == 1:
            for j in range(crc_length + 1):
                if i + j < len(msg):
                    msg[i + j] ^= (poly >> (crc_length - j)) & 1
    return np.array(msg[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_poly_divide(padded, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    remainder = _crc_poly_divide(bits, crc_length)
    return np.all(remainder == 0)


class _Path:
    """单条 SCL 路径"""
    __slots__ = ('L', 'B', 'pm', 'N', 'n')

    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def fork(self):
        p = object.__new__(_Path)
        p.N = self.N
        p.n = self.n
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L_size = self.list_size

        paths = [_Path(N, n, llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    pen = self._path_metric_penalty(llr, 0)
                    path.pm += pen
                    path.B[l, n] = 0
                    _update_bits(path.B, l, n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = path.fork()
                        pen = self._path_metric_penalty(llr, bit)
                        p.pm += pen
                        p.B[l, n] = bit
                        _update_bits(p.B, l, n)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:L_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, n].astype(int).copy(), best.pm
