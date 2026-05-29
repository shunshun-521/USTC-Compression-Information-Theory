"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _hard_decision,
)

# CRC-8: 0x07, CRC-16: 0x8005
_CRC_POLY = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = _CRC_POLY[crc_length]
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    rem = _crc_remainder(bits[:-crc_length], crc_length)
    expected = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat", "active")

  def __init__(self, N, n):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=int)
    self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组引用）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs_path(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = f_operation(
                        np.array([top_llr]), np.array([btm_llr])
                    )[0]
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        np.array([top_llr]), np.array([btm_llr]), np.array([top_bit])
                    )[0]

    def _update_bits_path(self, path, l):
        B = path.B
        n, N = self.n, self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u_bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs_path(path, l)
                llr_leaf = path.L[l, n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_leaf, 0)
                    path.pm += pen
                    path.B[l, n] = 0
                    path.u_hat[l] = 0
                    self._update_bits_path(path, l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(N, n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr_leaf, u_bit)
                        child.u_hat = path.u_hat.copy()
                        child.B[l, n] = u_bit
                        child.u_hat[l] = u_bit
                        self._update_bits_path(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        # 选择最优路径
        best = None
        if self.crc_length > 0:
            info_idx = self.info_indices
            for p in paths:
                info_bits = p.u_hat[info_idx]
                if crc_check(info_bits, self.crc_length):
                    if best is None or p.pm < best.pm:
                        best = p
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
