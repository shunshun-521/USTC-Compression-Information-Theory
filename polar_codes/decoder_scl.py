"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed, bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_lfsr(bits, poly, crc_length):
    """LFSR CRC 余数计算"""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << crc_length
    for b in bits:
        reg = ((reg << 1) | int(b)) & ((1 << (crc_length + 1)) - 1)
        if reg & top:
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_lfsr(padded, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_lfsr(bits, poly, crc_length) == 0


class _SCLPath:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_init
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)

    def clone(self):
        p = _SCLPath.__new__(_SCLPath)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SC + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            sorted(set(range(N)) - self.frozen_set), dtype=int
        )
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
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
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_add(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_init = llr_ch[self.br]

        paths = [_SCLPath(self.N, self.n, llr_init)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            is_frozen = l in self.frozen_set
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr0 = path.L[l, self.n]

                if is_frozen:
                    new_path = path.clone()
                    new_path.pm = self._pm_add(path.pm, llr0, 0)
                    new_path.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.clone()
                        new_path.pm = self._pm_add(path.pm, llr0, u)
                        new_path.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

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
