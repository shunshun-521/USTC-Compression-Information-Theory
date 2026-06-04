"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed_index,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length 仅支持 8 或 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=int)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u_bit):
        """路径度量惩罚：与 LLR 硬判决不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _update_llrs(self, path, l):
        start_s = self.n - _active_llr_level(l, self.n)
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top = path.L[j, s]
                    btm = path.L[j + branch_size, s]
                    path.L[j, s + 1] = f_operation(top, btm)
                else:
                    btm = path.L[j, s]
                    top = path.L[j - branch_size, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(top, btm, top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        end_s = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n) for _ in range(1)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    u_bit = 0
                    new_pm = path.pm + self._pm_penalty(llr, u_bit)
                    new_path = _Path(self.N, self.n)
                    new_path.L[:] = path.L
                    new_path.B[:] = path.B
                    new_path.pm = new_pm
                    new_path.u_hat[:] = path.u_hat
                    new_path.u_hat[l] = u_bit
                    new_path.B[l, self.n] = u_bit
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_pm = path.pm + self._pm_penalty(llr, u_bit)
                        new_path = _Path(self.N, self.n)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.pm = new_pm
                        new_path.u_hat[:] = path.u_hat
                        new_path.u_hat[l] = u_bit
                        new_path.B[l, self.n] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        best = paths[0]

        if self.crc_length > 0:
            info_bits_all = best.u_hat[self.info_indices]
            for p in paths:
                bits = p.u_hat[self.info_indices]
                if crc_check(bits, self.crc_length):
                    if p.pm < best.pm or not crc_check(
                        best.u_hat[self.info_indices], self.crc_length
                    ):
                        best = p

        return best.u_hat.copy(), best.pm
