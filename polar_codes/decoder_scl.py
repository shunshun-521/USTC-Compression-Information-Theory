"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _prepare_channel_llr,
    _update_llrs,
    _update_bits,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    expected = _crc_remainder(bits[:-crc_length], crc_length)
    received = 0
    for bit in bits[-crc_length:]:
        received = (received << 1) | int(bit)
    return expected == received


class Path:
    """单条译码路径。"""

    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = Path(self.L.shape[0], self.L.shape[1] - 1)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        channel_llr = _prepare_channel_llr(llr_ch, self.N)

        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = channel_llr

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_p = path.copy()
                    new_p.pm += self._pm_penalty(llr_val, 0)
                    new_p.u_hat[l] = 0
                    new_p.B[l, self.n] = 0
                    _update_bits(new_p.B, l, self.n, self.N)
                    new_paths.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = path.copy()
                        new_p.pm += self._pm_penalty(llr_val, u)
                        new_p.u_hat[l] = u
                        new_p.B[l, self.n] = u
                        _update_bits(new_p.B, l, self.n, self.N)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
