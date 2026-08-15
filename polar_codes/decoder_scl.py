"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    _frozen_set_from_mask,
    _prepare_llr,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


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
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = _Path.__new__(_Path)
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _frozen_set_from_mask(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, paths, phi):
        n = self.n
        N = self.N
        for path in paths:
            for s in range(n - _active_llr_level(phi, n), n):
                block_size = 1 << (s + 1)
                branch_size = block_size // 2
                for j in range(phi, N, block_size):
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

    def _update_bits(self, path, phi):
        if phi < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(phi, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(phi, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr)]

        for phi in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(paths, phi)
            candidates = []

            if phi in self.frozen_set:
                for path in paths:
                    llr_val = path.L[phi, self.n]
                    if llr_val < 0:
                        path.pm += abs(llr_val)
                    path.u_hat[phi] = 0
                    path.B[phi, self.n] = 0
                    self._update_bits(path, phi)
                    candidates.append(path)
            else:
                for path in paths:
                    llr_val = path.L[phi, self.n]
                    for bit in (0, 1):
                        child = path.copy()
                        if bit == 0 and llr_val < 0:
                            child.pm += abs(llr_val)
                        elif bit == 1 and llr_val >= 0:
                            child.pm += abs(llr_val)
                        child.u_hat[phi] = bit
                        child.B[phi, self.n] = bit
                        self._update_bits(child, phi)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path
            if best_pm is None or path.pm < best_pm.pm:
                best_pm = path

        chosen = best_crc if best_crc is not None else best_pm
        return chosen.u_hat.copy(), chosen.pm
