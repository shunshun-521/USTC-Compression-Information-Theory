"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level
from encoder import bit_reversal_permutation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ('L', 'B', 'pm', 'active')

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.active = True

    def clone(self):
        p = Path(len(self.L), int(math.log2(len(self.L))))
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _update_llrs(self, paths, l):
        for path in paths:
            start_s = self.n - _active_llr_level(l, self.n)
            for s in range(start_s, self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s], path.L[j, s],
                            path.B[j - branch_size, s + 1]
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        end_s = self.n - _active_bit_level(l, self.n)
        for path in paths:
            for s in range(self.n, end_s, -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                        path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch.copy()

        for l in self.decode_order:
            self._update_llrs(paths, l)
            llr_val = paths[0].L[l, self.n]

            if self.frozen_bits[l]:
                for path in paths:
                    path.pm += self._pm_penalty(llr_val, 0)
                    path.B[l, self.n] = 0
                self._update_bits(paths, l)
            else:
                candidates = []
                for path in paths:
                    for u in (0, 1):
                        new_path = path.clone()
                        new_path.pm += self._pm_penalty(llr_val, u)
                        new_path.B[l, self.n] = u
                        candidates.append(new_path)

                for cp in candidates:
                    self._update_bits([cp], l)

                candidates.sort(key=lambda p: p.pm)
                paths = candidates[:self.list_size]

        valid_paths = paths
        if self.crc_length > 0:
            crc_valid = []
            for path in paths:
                info_bits = path.B[:, self.n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)
            if crc_valid:
                valid_paths = crc_valid

        best = min(valid_paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int).copy(), best.pm
