"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    _prepare_llr,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) & 0xFF) ^ poly
                else:
                    reg = (reg << 1) & 0xFF
            else:
                break
        if crc_length == 16:
            if reg & 0x8000:
                reg = ((reg << 1) & 0xFFFF) ^ poly
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            msb = reg & (1 << (crc_length - 1))
            reg = (reg << 1) & ((1 << crc_length) - 1)
            if msb:
                reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            msb = reg & (1 << (crc_length - 1))
            reg = (reg << 1) & ((1 << crc_length) - 1)
            if msb:
                reg ^= poly
    return reg == 0


class _Path:
  __slots__ = ('L', 'B', 'pm', 'u_hat', 'active')

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=int)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=int)
    self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, self.llr_start, self.bit_start, _ = precompute_sc_indices(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llr(self, path, l, idx):
        start_s = self.n - self.llr_start[idx]
        for s in range(start_s, self.n):
            block_size = 2 ** (s + 1)
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

    def _update_bits(self, path, l, idx):
        if l < self.N // 2:
            return
        start_b = self.n - self.bit_start[idx]
        for s in range(self.n, start_b, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for idx, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                self._update_llr(path, l, idx)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(llr, 0)
                    path.pm += penalty
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l, idx)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = self._clone_path(path)
                        p.pm += self._path_metric_penalty(llr, bit)
                        p.B[l, self.n] = bit
                        p.u_hat[l] = bit
                        self._update_bits(p, l, idx)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

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
