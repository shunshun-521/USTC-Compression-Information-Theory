"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed_index,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(info_bits, poly, crc_length):
    """标准 CRC 余数位（MSB first）。"""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    steps = 8 if crc_length == 8 else 16
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(steps):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc = _crc_bits(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    steps = 8 if crc_length == 8 else 16
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(steps):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_proc):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_proc
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
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _path_metric_update(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        return pm if bit == hard else pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_proc = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr_proc)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    child = _Path(self.N, self.n, llr_proc)
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = self._path_metric_update(path.pm, llr, 0)
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, llr_proc)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = self._path_metric_update(path.pm, llr, bit)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm
