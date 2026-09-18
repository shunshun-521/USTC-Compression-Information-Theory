"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _bit_reversed,
    _prepare_llr,
    _update_bits,
    _update_llrs,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _int_to_bits(value, length):
    return np.array([(value >> (length - 1 - i)) & 1 for i in range(length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    register = 0
    mask = (1 << crc_length) - 1

    for bit in info_bits:
        register = ((register << 1) | int(bit)) & mask
        if register & (1 << (crc_length - 1)):
            register = ((register << 1) ^ poly) & mask

    for _ in range(crc_length):
        register = (register << 1) & mask
        if register & (1 << (crc_length - 1)):
            register = ((register << 1) ^ poly) & mask

    crc_bits = _int_to_bits(register, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits 是否满足 CRC 约束。
    """
    poly = CRC_POLYNOMIALS[crc_length]
    bits = np.asarray(bits, dtype=int)
    register = 0
    mask = (1 << crc_length) - 1

    for bit in bits:
        register = ((register << 1) | int(bit)) & mask
        if register & (1 << (crc_length - 1)):
            register = ((register << 1) ^ poly) & mask

    return register == 0


class Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)

    def copy(self):
        new_path = Path(self.L.shape[1] - 1, len(self.u_hat))
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        return new_path


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length

    @staticmethod
    def _path_metric_update(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = _prepare_llr(llr_ch)
        paths = [Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm = self._path_metric_update(path.pm, llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm = self._path_metric_update(child.pm, llr, bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n)
                        candidates.append(child)

            candidates.sort(key=lambda item: item.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda item: item.pm)
        else:
            best = min(paths, key=lambda item: item.pm)

        return best.u_hat.copy(), best.pm
