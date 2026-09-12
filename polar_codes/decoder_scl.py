"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _prepare_channel_llr,
    _update_bits,
    _update_llrs,
    bit_reversed,
    hard_decision,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) | int(bit)
        if msb ^ int(bit):
            reg ^= poly

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _SCLPath:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（Vangala 置换 SC + 路径度量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_llr(self, path, l):
        _update_llrs(path.L, path.B, l, self.n)
        return path.L[l, self.n]

    def _continue_path(self, path, l, u_bit):
        new_path = _SCLPath(self.N, self.n, path.L[:, 0].copy())
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.B[l, self.n] = u_bit
        _update_bits(new_path.B, l, self.n)
        return new_path

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        paths = [_SCLPath(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            is_frozen = l in self.frozen_set
            candidates = []

            for path in paths:
                llr_val = self._path_llr(path, l)
                if is_frozen:
                    pm = path.pm + (0.0 if llr_val >= 0 else abs(llr_val))
                    new_path = self._continue_path(path, l, 0)
                    new_path.pm = pm
                    candidates.append((pm, new_path))
                else:
                    for u_bit, penalty_sign in ((0, llr_val >= 0), (1, llr_val < 0)):
                        pm = path.pm + (0.0 if penalty_sign else abs(llr_val))
                        new_path = self._continue_path(path, l, u_bit)
                        new_path.pm = pm
                        candidates.append((pm, new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [item[1] for item in candidates[: self.list_size]]

        if self.crc_length > 0:
            valid = []
            frozen_mask = np.array([i in self.frozen_set for i in range(self.N)], dtype=bool)
            for i, path in enumerate(paths):
                info_bits = path.B[:, self.n][~frozen_mask]
                if crc_check(info_bits, self.crc_length):
                    valid.append((path.pm, i))
            if valid:
                best = paths[min(valid, key=lambda x: x[0])[1]]
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm
