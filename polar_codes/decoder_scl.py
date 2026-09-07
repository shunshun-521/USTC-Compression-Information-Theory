"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    lower_llr,
    upper_llr_exact,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class Path:
    """SCL 单条路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.active = True

    def copy(self):
        p = Path.__new__(Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.active = True
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, use_min_sum=False):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.use_min_sum = use_min_sum
        self.f_func = f_operation if use_min_sum else upper_llr_exact
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = self.f_func(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if hard == bit else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        if self.list_size == 1:
            from decoder_sc import sc_decode_core
            u_hat = sc_decode_core(llr_ch, self.frozen_bits, use_min_sum=self.use_min_sum)
            return u_hat, 0.0

        paths = [Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    p0 = path.copy()
                    p0.pm += self._path_metric_penalty(llr_val, 0)
                    p0.B[l, self.n] = 0
                    self._update_bits(p0, l)
                    new_paths.append(p0)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        p.pm += self._path_metric_penalty(llr_val, bit)
                        p.B[l, self.n] = bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            candidates.append((path.pm, u_hat))

        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[info_pos], self.crc_length)
            ]
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], candidates[0][0]
