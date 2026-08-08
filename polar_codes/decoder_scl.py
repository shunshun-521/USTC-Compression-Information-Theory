"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _bit_reversed_index, _update_llr, _update_bits


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    data = bits[:-crc_length]
    rem = _crc_remainder(data, poly, crc_length)
    expected = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================

class _Path:
    """单条译码路径"""

    def __init__(self, N, n, llr):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, n] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """
    SCL 译码器（路径数组实现，列表大小较小时足够高效）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr_val, bit):
        """不一致分支加 |LLR| 惩罚"""
        hard = 0 if llr_val >= 0 else 1
        return abs(llr_val) if bit != hard else 0.0

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        br = bit_reversal_permutation(self.N)
        llr = np.asarray(llr_ch, dtype=np.float64)[br]

        paths = [_Path(self.N, self.n, llr.copy())]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []

            for path in paths:
                _update_llr(path.L, path.B, l, self.n)

                if self.frozen_bits[l]:
                    penalty = 0.0 if path.L[l, 0] >= 0 else abs(path.L[l, 0])
                    new_path = _Path(self.N, self.n, llr)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat = path.u_hat.copy()
                    new_path.pm = path.pm + penalty
                    new_path.B[l, 0] = 0
                    new_path.u_hat[l] = 0
                    _update_bits(new_path.B, l, self.n)
                    new_paths.append(new_path)
                else:
                    llr_leaf = path.L[l, 0]
                    for bit in (0, 1):
                        new_path = _Path(self.N, self.n, llr)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.pm = path.pm + self._path_metric_penalty(llr_leaf, bit)
                        new_path.B[l, 0] = bit
                        new_path.u_hat[l] = bit
                        _update_bits(new_path.B, l, self.n)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
