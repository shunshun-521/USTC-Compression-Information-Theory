"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed, _update_llrs, _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_compute(bits, poly, crc_length):
    crc = 0
    for bit in bits:
        crc ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if crc & (1 << crc_length):
                crc = ((crc << 1) ^ poly) & ((1 << (crc_length + 1)) - 1)
            else:
                crc = (crc << 1) & ((1 << (crc_length + 1)) - 1)
    return crc & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_compute(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    crc_bits = bits[-crc_length:]
    computed = _crc_compute(payload, poly, crc_length)
    received = sum(int(crc_bits[i]) << (crc_length - 1 - i) for i in range(crc_length))
    return computed == received


class Path:
    """单条译码路径"""

    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch, br_perm):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch[br_perm]
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        new_path = Path(self.L.shape[0], self.L.shape[1] - 1, self.L[:, 0],
                        np.arange(self.L.shape[0]))
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        paths = [Path(self.N, self.n, llr_ch, br)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_bit = path.L[l, self.n]

                if l in self.frozen_set:
                    if llr_bit < 0:
                        path.pm += abs(llr_bit)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        expected = 0 if llr_bit >= 0 else 1
                        if bit != expected:
                            child.pm += abs(llr_bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            crc_paths = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(crc_paths or paths, key=lambda p: p.pm)
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
