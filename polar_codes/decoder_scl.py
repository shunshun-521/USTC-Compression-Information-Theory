"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    apply_llr_deperm,
    _active_bit_level,
    _active_llr_level,
    bit_reversed_index,
    f_operation,
    g_operation,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_update(reg, bit, poly, crc_length):
    reg ^= bit << (crc_length - 1)
    mask = 1 << (crc_length - 1)
    full_mask = (1 << crc_length) - 1
    if reg & mask:
        reg = ((reg << 1) ^ poly) & full_mask
    else:
        reg = (reg << 1) & full_mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, int(bit), poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, int(bit), poly, crc_length)
    return reg == 0


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _SCLPath:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """置换 SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(self.N)]

    def _clone(self, path):
        new_path = _SCLPath(self.N, self.n)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = apply_llr_deperm(np.asarray(llr_ch, dtype=np.float64))
        paths = [_SCLPath(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen:
                    new_path = self._clone(path)
                    new_path.pm = _pm_update(new_path.pm, cur_llr, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._clone(path)
                        new_path.pm = _pm_update(new_path.pm, cur_llr, u)
                        new_path.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat.copy(), path.pm

        best = paths[0]
        return best.u_hat.copy(), best.pm
