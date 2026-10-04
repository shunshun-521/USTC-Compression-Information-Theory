"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import math
import numpy as np
from encoder import bit_reversed_index
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level, _update_llrs, _update_bits


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_bits(data_bits, poly, crc_len):
    """对 data_bits（含尾部 crc_len 个 0）计算 CRC 余数位。"""
    reg = 0
    mask = (1 << crc_len) - 1
    top = 1 << (crc_len - 1)
    for b in data_bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array([(reg >> i) & 1 for i in range(crc_len - 1, -1, -1)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    crc_bits = _crc_bits(padded, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem_bits = _crc_bits(bits, poly, crc_length)
    return np.all(rem_bits == 0)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径复制在分裂时进行，列表规模为 L）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_llr_at(self, path, l):
        _update_llrs(path.L, path.B, l, self.n)
        return path.L[l, self.n]

    def _path_set_bit(self, path, l, bit, llr_val):
        path.u_hat[l] = bit
        path.B[l, self.n] = bit
        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0) else abs(llr_val)
        path.pm += penalty
        _update_bits(path.B, l, self.n)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for p in paths:
                if not p.active:
                    continue
                llr_val = self._path_llr_at(p, l)
                if self.frozen_bits[l]:
                    new_p = self._clone_path(p)
                    self._path_set_bit(new_p, l, 0, llr_val)
                    candidates.append(new_p)
                else:
                    for bit in (0, 1):
                        new_p = self._clone_path(p)
                        self._path_set_bit(new_p, l, bit, llr_val)
                        candidates.append(new_p)

            candidates.sort(key=lambda x: x.pm)
            paths = candidates[: self.list_size]

        best = self._select_path(paths)
        return best.u_hat.copy(), best.pm

    def _clone_path(self, path):
        new_p = _Path(self.N, self.n, path.L[:, 0])
        new_p.L = path.L.copy()
        new_p.B = path.B.copy()
        new_p.pm = path.pm
        new_p.u_hat = path.u_hat.copy()
        return new_p

    def _select_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p.u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda x: x.pm)
        return min(paths, key=lambda x: x.pm)
