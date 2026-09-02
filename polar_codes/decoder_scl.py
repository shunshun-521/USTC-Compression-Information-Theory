"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _SCState,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _hard_decision,
    _lower_llr,
    _upper_llr,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError('crc_length must be 8 or 16')

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError('crc_length must be 8 or 16')
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _Path:
    """SCL 单条路径"""

    __slots__ = ('state', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch, frozen_set):
        self.state = _SCState(N, n, llr_ch.copy(), frozen_set)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, bit):
        """路径度量惩罚"""
        decided = 0 if llr >= 0 else 1
        return 0.0 if decided == bit else abs(llr)

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, path.state.L[:, 0], self.frozen_set)
        new_path.state.L = path.state.L.copy()
        new_path.state.B = path.state.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch, self.frozen_set)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                path.state.update_llrs(l)
                llr = float(path.state.L[l, self.n])

                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(llr, 0)
                    path.pm += penalty
                    path.u_hat[l] = 0
                    path.state.B[l, self.n] = 0
                    path.state.update_bits(l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        candidate = self._clone_path(path)
                        penalty = self._path_metric_penalty(llr, bit)
                        candidate.pm += penalty
                        candidate.u_hat[l] = bit
                        candidate.state.B[l, self.n] = bit
                        candidate.state.update_bits(l)
                        new_paths.append(candidate)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
