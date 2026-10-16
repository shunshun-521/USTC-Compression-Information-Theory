"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation
from decoder_sc import (
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
    f_operation,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc8_remainder(bits):
    crc = 0
    for b in bits:
        crc ^= int(b) << 7
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ _CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_bits(bits):
    crc = 0
    for b in bits:
        crc ^= int(b) << 15
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        rem = _crc8_remainder(info_bits)
        crc_bits = np.array([(rem >> i) & 1 for i in range(7, -1, -1)], dtype=int)
    elif crc_length == 16:
        rem = _crc16_bits(info_bits)
        crc_bits = np.array([(rem >> i) & 1 for i in range(15, -1, -1)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        L, B = path.L, path.B
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        N, n = self.N, self.n
        B = path.B
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, u_bit):
        """不一致分支加 |LLR|"""
        v = 0 if llr >= 0 else 1
        return 0.0 if u_bit == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for step in range(self.N):
            l = self.br[step]
            candidates = []

            for p_idx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr_leaf = path.L[l, self.n]
                if self.frozen_bits[l]:
                    pen = self._path_metric_penalty(llr_leaf, 0)
                    new = _Path(self.N, self.n, llr_ch)
                    new.pm = path.pm + pen
                    new.L = path.L.copy()
                    new.B = path.B.copy()
                    new.u_hat = path.u_hat.copy()
                    new.u_hat[l] = 0
                    new.B[l, self.n] = 0
                    self._update_bits(new, l)
                    candidates.append(new)
                else:
                    for u_bit in (0, 1):
                        pen = self._path_metric_penalty(llr_leaf, u_bit)
                        new = _Path(self.N, self.n, llr_ch)
                        new.pm = path.pm + pen
                        new.L = path.L.copy()
                        new.B = path.B.copy()
                        new.u_hat = path.u_hat.copy()
                        new.u_hat[l] = u_bit
                        new.B[l, self.n] = u_bit
                        self._update_bits(new, l)
                        candidates.append(new)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        # 选择最优路径
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            pool = valid if valid else paths
        else:
            pool = paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
