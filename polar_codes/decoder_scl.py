"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, precompute_sc_indices, _frozen_to_set,
    _active_llr_level, _active_bit_level, _bit_reversed,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    """SCL 单条路径（Lazy Copy）"""

    __slots__ = ('L', 'B', 'pm', 'parent', 'copy_L', 'copy_B')

    def __init__(self, N, n, llr_ch, parent=None):
        self.parent = parent
        self.copy_L = parent is None
        self.copy_B = parent is None
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.L[:, 0] = llr_ch
        else:
            self.L = parent.L
            self.B = parent.B
        self.pm = parent.pm if parent else 0.0

    def ensure_L_copy(self):
        if not self.copy_L:
            self.L = self.L.copy()
            self.copy_L = True

    def ensure_B_copy(self):
        if not self.copy_B:
            self.B = self.B.copy()
            self.copy_B = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_to_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, self.llr_start_layer, self.bit_start_layer = precompute_sc_indices(N)

        if crc_length > 0:
            info_positions = sorted(set(range(N)) - self.frozen_set)
            self.crc_positions = set(info_positions[-crc_length:])
            self.payload_positions = set(info_positions[:-crc_length])
        else:
            self.crc_positions = set()
            self.payload_positions = set(range(N)) - self.frozen_set

    def _update_llrs(self, path, l, idx):
        start_s = self.n - self.llr_start_layer[idx]
        for s in range(start_s, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    val = f_operation(path.L[j, s], path.L[j + branch_size, s])
                    if path.L[j, s + 1] != val:
                        path.ensure_L_copy()
                        path.L[j, s + 1] = val
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    val = g_operation(path.L[j - branch_size, s], path.L[j, s], top_bit)
                    if path.L[j, s + 1] != val:
                        path.ensure_L_copy()
                        path.L[j, s + 1] = val

    def _update_bits(self, path, l, idx):
        if l < self.N // 2:
            return
        start_bit_s = self.n - self.bit_start_layer[idx]
        for s in range(self.n, start_bit_s, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.ensure_B_copy()
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for idx, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                self._update_llrs(path, l, idx)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    p0 = Path(self.N, self.n, llr_ch, parent=path)
                    p0.pm = path.pm + self._pm_penalty(llr, 0)
                    p0.ensure_B_copy()
                    p0.B[l, self.n] = 0
                    self._update_bits(p0, l, idx)
                    new_paths.append(p0)
                else:
                    for bit in (0, 1):
                        p = Path(self.N, self.n, llr_ch, parent=path)
                        p.pm = path.pm + self._pm_penalty(llr, bit)
                        p.ensure_B_copy()
                        p.B[l, self.n] = bit
                        self._update_bits(p, l, idx)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        best_crc = None
        best_all = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            for p in paths:
                u = p.B[:, self.n].astype(int)
                info_bits = [u[i] for i in sorted(self.payload_positions | self.crc_positions)]
                if crc_check(np.array(info_bits, dtype=int), self.crc_length):
                    if best_crc is None or p.pm < best_crc.pm:
                        best_crc = p

        chosen = best_crc if best_crc is not None else best_all
        return chosen.B[:, self.n].astype(int), chosen.pm
