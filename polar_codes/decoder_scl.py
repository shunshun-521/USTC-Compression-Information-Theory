"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _compute_llr, _b_check, _s_updater


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ('llrs', 's', 'pm', 'u_hat')

    def __init__(self, N, n):
        self.llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
        self.s = np.full((n + 1, N), -1, dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _copy_path(self, src):
        dst = _Path(self.N, self.n)
        dst.llrs = src.llrs.copy()
        dst.s = src.s.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].llrs[self.n, :] = llr_ch

        for phi in range(self.N):
            new_paths = []

            if self.frozen_bits[phi]:
                for path in paths:
                    new_path = self._copy_path(path)
                    new_path.llrs[0, phi] = np.inf
                    new_path.s[0, phi] = 0
                    new_path.u_hat[phi] = 0
                    new_path.pm += self._pm_penalty(
                        _compute_llr(0, phi, new_path.llrs, new_path.s, self.n), 0
                    )
                    new_paths.append(new_path)
            else:
                for path in paths:
                    llr = _compute_llr(0, phi, path.llrs, path.s, self.n)
                    for u in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.llrs[0, phi] = llr
                        new_path.s[0, phi] = u
                        new_path.u_hat[phi] = u
                        new_path.pm += self._pm_penalty(llr, u)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[~self.frozen_bits], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
