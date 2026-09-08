"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed,
    _active_llr_level, _active_bit_level, _hard_decision,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _compute_crc_bits(info_bits, crc_length):
    """计算 CRC 比特（自洽校验和）"""
    val = 0
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    for i, bit in enumerate(info_bits):
        val = ((val << 1) | int(bit)) & ((1 << crc_length) - 1)
        if val & (1 << (crc_length - 1)):
            val ^= poly
    return np.array(
        [(val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _compute_crc_bits(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = _compute_crc_bits(info, crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class PathState:
    """SCL 单条路径状态"""

    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int32)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, state, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    state.L[j, s + 1] = f_operation(state.L[j, s], state.L[j + branch_size, s])
                else:
                    state.L[j, s + 1] = g_operation(
                        state.L[j - branch_size, s],
                        state.L[j, s],
                        state.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, state, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    state.B[j - branch_size, s - 1] = int(state.B[j, s]) ^ int(state.B[j - branch_size, s])
                    state.B[j, s - 1] = state.B[j, s]

    def _crc_pass(self, u_hat):
        info_positions = np.where(~self.frozen_bits)[0]
        return crc_check(u_hat[info_positions], self.crc_length)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    child = copy.copy(path)
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = path.pm + penalty
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0) else abs(llr_val)
                        child = copy.copy(path)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + penalty
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            passed = [p for p in paths if self._crc_pass(p.u_hat)]
            if passed:
                best = min(passed, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
