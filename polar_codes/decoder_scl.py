"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _scatter_channel_llrs,
    _update_bits,
    _update_llrs,
    f_operation,
    path_metric_penalty,
    sc_decode,
)
from encoder import bit_reversal_permutation

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_len - 1)
        for _ in range(crc_len):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    """单条译码路径（Lazy Copy：仅在分裂时复制数组）。"""

    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch, rev):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = _scatter_channel_llrs(llr_ch, rev)

    def clone(self):
        child = _Path.__new__(_Path)
        child.pm = self.pm
        child.L = self.L.copy()
        child.B = self.B.copy()
        return child


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self._info_idx = np.where(~self.frozen_bits)[0]
        self._rev = bit_reversal_permutation(N)
        self._decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [_Path(self.N, self.n, llr_ch, self._rev)]

        for l in self._decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]
                bits = [0] if self.frozen_bits[l] else [0, 1]
                for bit in bits:
                    child = path.clone()
                    child.pm += path_metric_penalty(llr, bit)
                    child.B[l, self.n] = bit
                    _update_bits(child.B, l, self.n, self.N)
                    candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_paths = [
                p
                for p in paths
                if crc_check(p.B[:, self.n][self._info_idx], self.crc_length)
            ]
            pool = crc_paths if crc_paths else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
