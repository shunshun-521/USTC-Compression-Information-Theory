"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level
from encoder import bit_reversed


# CRC-8: x^8 + x^2 + x + 1 (0x07)
_CRC8_POLY = 0x07
# CRC-16: 0x8005
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC 只覆盖 info_bits，不含冻结位。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _Path:
    """单条 SCL 路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "parent")

    def __init__(self, N, n, llr_ch, parent=None):
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.L[:, 0] = llr_ch
            self.pm = 0.0
            self.parent = None
        else:
            self.L = parent.L
            self.B = parent.B
            self.pm = parent.pm
            self.parent = parent


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, None, parent=path)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.parent = None
        return new_path

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        L, B = path.L, path.B
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top, btm, top_bit)

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        B = path.B
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, bit):
        """与 LLR 不一致时加 |LLR| 惩罚"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        decode_order = [bit_reversed(i, n) for i in range(N)]

        paths = [_Path(N, n, llr_ch)]

        for l in decode_order:
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_leaf = path.L[l, n]

                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(llr_leaf, 0)
                    new_path = self._clone_path(path)
                    new_path.pm += penalty
                    new_path.B[l, n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._path_metric_penalty(llr_leaf, bit)
                        new_path.B[l, n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        # 选择最优路径（CRC 辅助）
        best_path = paths[0]
        best_pm = paths[0].pm
        u_hat = paths[0].B[:, n].copy()

        if self.crc_length > 0:
            crc_pass = []
            for path in paths:
                decoded_info = path.B[self.info_indices, n]
                if crc_check(decoded_info, self.crc_length):
                    crc_pass.append(path)
            if crc_pass:
                best_path = min(crc_pass, key=lambda p: p.pm)
                u_hat = best_path.B[:, n].copy()
                best_pm = best_path.pm

        return u_hat, best_pm
