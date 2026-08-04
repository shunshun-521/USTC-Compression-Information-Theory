"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _update_llrs,
    _update_bits,
    LLR_CLIP,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


# ==================== SCL 译码器 ====================


class _SCLPath:
    """单条 SCL 译码路径"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0

    def copy(self):
        p = _SCLPath(self.N, self.n)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


def _update_llrs_path(L, B, l, n, N):
    """更新单条路径的 LLR（与 SC 相同）"""
    _update_llrs(L, B, l, n, N)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_update(self, pm, llr, bit):
        """路径度量更新"""
        if bit == 0:
            penalty = 0.0 if llr >= 0 else abs(llr)
        else:
            penalty = 0.0 if llr < 0 else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """主译码函数"""
        N = self.N
        n = self.n

        paths = []
        p0 = _SCLPath(N, n)
        p0.L[:, 0] = np.clip(llr_ch, -LLR_CLIP, LLR_CLIP)
        paths.append(p0)

        decode_order = [_bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            candidates = []

            for path in paths:
                _update_llrs_path(path.L, path.B, l, n, N)
                llr = path.L[l, n]
                if np.isnan(llr):
                    llr = 0.0

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.B[l, n] = 0
                    new_path.pm = self._path_metric_update(path.pm, llr, 0)
                    _update_bits(new_path.B, l, n, N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.B[l, n] = bit
                        new_path.pm = self._path_metric_update(path.pm, llr, bit)
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        u_hat = paths[0].B[:, n].astype(np.int8)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p.B[:, n].astype(np.int8)
                info_and_crc = bits[self.info_indices]
                if crc_check(info_and_crc, self.crc_length):
                    valid.append(p)
            if valid:
                valid.sort(key=lambda p: p.pm)
                u_hat = valid[0].B[:, n].astype(np.int8)

        return u_hat, paths[0].pm
