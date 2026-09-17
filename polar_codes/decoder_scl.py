"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)

# ==================== CRC 工具 ====================

_POLY8 = [1, 0, 0, 0, 0, 0, 1, 1, 1]  # x^8 + x^2 + x + 1 (0x07)
_POLY16 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1]  # 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _POLY8
    if crc_length == 16:
        return _POLY16
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(data_bits, poly_bits, crc_len):
    reg = list(map(int, data_bits)) + [0] * crc_len
    n = len(data_bits)
    for i in range(n):
        if reg[i]:
            for j in range(len(poly_bits)):
                if i + j < len(reg):
                    reg[i + j] ^= poly_bits[j]
    return np.array(reg[n:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    remainder = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = list(map(int, bits))
    n = len(bits) - crc_length
    for i in range(n):
        if reg[i]:
            for j in range(len(poly)):
                if i + j < len(reg):
                    reg[i + j] ^= poly[j]
    return all(x == 0 for x in reg[n:])


# ==================== SCL 译码器 ====================


class _Path:
    """单条译码路径（Lazy Copy：共享未修改的 LLR/比特数组）。"""

    __slots__ = ("L", "B", "pm", "u_hat", "owner_id")

    def __init__(self, N, n, llr_ch, owner_id):
        self.owner_id = owner_id
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self, new_id):
        p = _Path.__new__(_Path)
        p.owner_id = new_id
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

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
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        paths = [_Path(self.N, self.n, llr_ch, 0)]
        next_id = 1

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    bit = 0
                    new_path = path.copy(next_id)
                    next_id += 1
                    new_path.pm += self._pm_penalty(llr, bit)
                    new_path.B[l, self.n] = bit
                    new_path.u_hat[l] = bit
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy(next_id)
                        next_id += 1
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        new_path.u_hat[l] = bit
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
