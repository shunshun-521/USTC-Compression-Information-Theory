"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_division(data_bits, poly, crc_length):
    """多项式长除法计算 CRC"""
    reg = [0] * crc_length
    for bit in data_bits:
        feedback = bit ^ reg[0]
        reg = reg[1:] + [0]
        for i in range(crc_length):
            if feedback and ((poly >> (crc_length - 1 - i)) & 1):
                reg[i] ^= 1
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc_bits = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_division(bits, poly, crc_length)
    return np.all(remainder == 0)


# ==================== SCL 译码器 ====================


class _Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True

    def copy(self):
        new_path = _Path.__new__(_Path)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.active = True
        return new_path


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_set = set(np.where(~self.frozen_bits)[0])

    def _path_metric_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]

        for l in decode_order:
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                _update_llrs(path.L, path.B, l, self.n)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._path_metric_penalty(llr_val, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr_val, bit)
                        new_path.B[l, self.n] = bit
                        _update_bits(new_path.B, l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        # 选择最优路径
        crc_pass = [
            p for p in paths
            if self.crc_length == 0 or self._check_crc(p.B[:, self.n])
        ]
        best = min(crc_pass if crc_pass else paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm

    def _check_crc(self, u_hat):
        info_bits = u_hat[sorted(self.info_set)]
        return crc_check(info_bits, self.crc_length)
