"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _prepare_channel_llr,
    _frozen_indices,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _f_boxplus,
    _g_boxplus,
    g_operation,
)
from utils import crc_encode as _crc_encode_util, crc_check as _crc_check_util


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    return _crc_encode_util(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    return _crc_check_util(bits, crc_length)


class _Path:
    """单条 SCL 路径，使用 lazy copy 索引"""

    __slots__ = ("L", "B", "pm", "u_hat", "parent", "copied")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.parent = None
        self.copied = False

    def copy(self):
        child = _Path.__new__(_Path)
        child.L = self.L
        child.B = self.B
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        child.parent = self
        child.copied = False
        return child

    def ensure_copy(self):
        if not self.copied and self.parent is not None:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.copied = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_indices(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = sorted(set(range(N)) - self.frozen_set)

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _g_boxplus(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path.B
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        decode_order = [_bit_reversed_index(i, self.n) for i in range(self.N)]

        for l in decode_order:
            new_paths = []

            for path in paths:
                path.ensure_copy()
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.ensure_copy()
                        child.pm += self._pm_penalty(llr, bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_pass(p.u_hat)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _crc_pass(self, u_hat):
        info_bits = u_hat[self.info_indices]
        if self.crc_length <= 0:
            return True
        k_info = len(self.info_indices) - self.crc_length
        payload = info_bits[:k_info]
        received = info_bits[k_info:]
        expected = _crc_encode_util(payload, self.crc_length)[k_info:]
        return np.array_equal(received, expected)
