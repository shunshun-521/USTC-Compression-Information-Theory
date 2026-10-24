"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _lower_llr,
    _upper_llr,
    _frozen_array,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if bits.size < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _path_metric_update(pm, llr, u):
    """路径度量：与 LLR 不一致时加 |LLR|"""
    if u == (0 if llr >= 0 else 1):
        return pm
    return pm + abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_pc):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_pc
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径列表）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = _frozen_array(frozen_bits)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone_path(self, path):
        new = _Path(self.N, self.n, path.L[:, 0].copy())
        new.L[:] = path.L
        new.B[:] = path.B
        new.pm = path.pm
        new.u_hat = path.u_hat.copy()
        return new

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_pc = llr_ch[br]

        paths = [_Path(self.N, self.n, llr_pc)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pm = _path_metric_update(path.pm, llr, 0)
                    new = self._clone_path(path)
                    new.pm = pm
                    new.B[l, self.n] = 0
                    new.u_hat[l] = 0
                    self._update_bits(new, l)
                    candidates.append(new)
                else:
                    for u in (0, 1):
                        pm = _path_metric_update(path.pm, llr, u)
                        new = self._clone_path(path)
                        new.pm = pm
                        new.B[l, self.n] = u
                        new.u_hat[l] = u
                        self._update_bits(new, l)
                        candidates.append(new)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        # 选择输出路径
        valid = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            else:
                valid.append(p)

        pool = valid if valid else paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
