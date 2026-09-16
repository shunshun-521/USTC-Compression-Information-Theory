"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divide(info_bits, poly, crc_length):
    bits = np.concatenate([info_bits.astype(int), np.zeros(crc_length, dtype=int)])
    poly_bits = np.array([(poly >> i) & 1 for i in range(crc_length, -1, -1)], dtype=int)
    for i in range(len(info_bits)):
        if bits[i] == 1:
            bits[i:i + len(poly_bits)] ^= poly_bits
    return bits[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    check = _crc_divide(bits[:-crc_length], poly, crc_length)
    return np.array_equal(check, bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        return abs(llr) if u != u_from_llr else 0.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        br = np.array([bit_reversed(i, n) for i in range(N)], dtype=int)
        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch[br]

        for phi in range(N):
            l = bit_reversed(phi, n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr0 = path.L[l, n]

                if l in self.frozen_set:
                    new_path = _Path(N, n)
                    new_path.L[:] = path.L
                    new_path.B[:] = path.B
                    new_path.u_hat[:] = path.u_hat
                    new_path.pm = path.pm + self._path_metric_penalty(llr0, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, n] = 0
                    _update_bits(new_path.B, l, n)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = _Path(N, n)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.u_hat[:] = path.u_hat
                        new_path.pm = path.pm + self._path_metric_penalty(llr0, u_val)
                        new_path.u_hat[l] = u_val
                        new_path.B[l, n] = u_val
                        _update_bits(new_path.B, l, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_pass = []
        for p in paths:
            info_bits = p.u_hat[self.info_indices]
            if self.crc_length > 0 and crc_check(info_bits, self.crc_length):
                crc_pass.append(p)

        best = min(crc_pass if crc_pass else paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
