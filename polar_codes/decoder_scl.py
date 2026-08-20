"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(info_bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= poly
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
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
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "copied")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.copied = False

    def copy(self):
        new_path = _Path(len(self.u_hat), int(np.log2(len(self.u_hat))))
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        new_path.copied = True
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _metric_update(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        br = np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)
        llr_ch = llr_ch[br]

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    child = path.copy()
                    child.pm = self._metric_update(path.pm, llr, 0)
                    child.B[l, n] = 0
                    child.u_hat[l] = 0
                    _update_bits(child.B, l, n, N)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm = self._metric_update(path.pm, llr, bit)
                        child.B[l, n] = bit
                        child.u_hat[l] = bit
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
