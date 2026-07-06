"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _update_llrs,
    _update_bits,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    return _CRC8_POLY if crc_length == 8 else _CRC16_POLY


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1

    for bit in info_bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1

    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if feedback:
            reg ^= poly

    return reg == 0


class _Path:
    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        br = bit_reversal_permutation(N)
        self.L[:, 0] = llr_ch[br]


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pm = path.pm + self._pm_penalty(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._pm_penalty(llr, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in survivors:
                child = _Path(self.N, self.n, llr_ch)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.u_hat[l] = bit
                child.B[l, self.n] = bit
                _update_bits(child.B, l, self.n, self.N)
                new_paths.append(child)

            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            pool = valid if valid else paths
            best = min(pool, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
