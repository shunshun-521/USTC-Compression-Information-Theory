"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    prepare_channel_llr,
    _active_bit_level,
    _active_llr_level,
)
from encoder import bit_reversed


# ==================== CRC 工具 ====================

# CRC 多项式标识（供文档引用）
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


_CRC8_GEN = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)   # x^8+x^2+x+1
_CRC16_GEN = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int)


def _gf2_remainder(msg, gen):
    msg = list(map(int, msg))
    gen = list(map(int, gen))
    while len(msg) >= len(gen):
        if msg[0] == 1:
            for i in range(len(gen)):
                msg[i] ^= gen[i]
        msg = msg[1:]
    return np.array(msg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _gf2_remainder(msg, gen)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    gen = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    rem = _gf2_remainder(bits, gen)
    return np.all(rem == 0)


# ==================== SCL 译码器 ====================

class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _path_metric_update(self, pm, llr, bit):
        """路径度量更新。"""
        if bit == 0:
            penalty = 0.0 if llr >= 0 else abs(llr)
        else:
            penalty = 0.0 if llr < 0 else abs(llr)
        return pm + penalty

    def _update_llrs(self, path, l):
        start = self.n - _active_llr_level(l, self.n)
        for s in range(start, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        end = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        llr_ch: 自然顺序信道 LLR（内部会比特倒序）。
        """
        llr_ch = prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi_idx, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pm = self._path_metric_update(path.pm, llr, 0)
                    path.pm = pm
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        pcopy = _Path(self.N, self.n, llr_ch)
                        pcopy.pm = self._path_metric_update(path.pm, llr, bit)
                        pcopy.L = path.L.copy()
                        pcopy.B = path.B.copy()
                        pcopy.u_hat = path.u_hat.copy()
                        pcopy.B[l, self.n] = bit
                        pcopy.u_hat[l] = bit
                        self._update_bits(pcopy, l)
                        new_paths.append(pcopy)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_positions = np.where(self.frozen_bits == 0)[0]
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[info_positions], self.crc_length)
            ]
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
