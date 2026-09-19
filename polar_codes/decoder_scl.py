"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _reorder_channel_llrs,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0b100000111      # x^8 + x^2 + x + 1
_CRC16_POLY = 0b10000000000000101  # x^16 + x^15 + x^2 + 1 (0x8005)


def _crc_mod(bits, poly, crc_length):
    """MSB-first CRC 模运算（移位寄存器实现）"""
    top = 1 << crc_length
    mask = (1 << (crc_length + 1)) - 1
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & top:
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。

    使用标准多项式：
      r=8:  CRC-8  (0x07, 即 x^8 + x^2 + x + 1)
      r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_mod(padded, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    return _crc_mod(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================

class _Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)
        self.decode_order = np.array(
            [_bit_reversed_index(i, self.n) for i in range(N)], dtype=int
        )
        self.llr_start_layer = np.array(
            [self.n - _active_llr_level(l, self.n) for l in self.decode_order], dtype=int
        )
        self.bit_start_layer = np.array(
            [self.n - _active_bit_level(l, self.n) for l in self.decode_order], dtype=int
        )

    def _copy_path(self, src):
        dst = _Path(self.N, self.n)
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def _update_llrs(self, path, l, idx):
        for s in range(self.llr_start_layer[idx], self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l, idx):
        if l < self.N // 2:
            return
        for s in range(self.n, self.bit_start_layer[idx], -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列（最优路径）
            pm: 最优路径的度量值
        """
        llr_ch = _reorder_channel_llrs(np.asarray(llr_ch, dtype=np.float64))

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for idx, l in enumerate(self.decode_order):
            new_paths = []

            for path in paths:
                if not path.active:
                    continue

                self._update_llrs(path, l, idx)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._path_metric_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l, idx)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = self._copy_path(path)
                        child.pm += self._path_metric_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._update_bits(child, l, idx)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
