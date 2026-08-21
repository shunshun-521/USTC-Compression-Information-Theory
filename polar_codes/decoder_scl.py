"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= poly

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class Path:
    """SCL 单条路径。"""

    __slots__ = ('L', 'B', 'pm', 'u_hat', 'N', 'n')

    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch.copy()
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = Path(self.N, self.n, self.L[:, 0])
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    @staticmethod
    def _hard_bit(llr):
        return 0 if llr >= 0 else 1

    def _path_metric_penalty(self, llr, u):
        return 0.0 if u == self._hard_bit(llr) else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _propagate_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._path_metric_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._propagate_bits(path, l)
                    new_candidates.append(path)
                else:
                    p0 = path.copy()
                    p0.pm += self._path_metric_penalty(llr, 0)
                    p0.B[l, self.n] = 0
                    p0.u_hat[l] = 0
                    self._propagate_bits(p0, l)
                    new_candidates.append(p0)

                    p1 = path.copy()
                    p1.pm += self._path_metric_penalty(llr, 1)
                    p1.B[l, self.n] = 1
                    p1.u_hat[l] = 1
                    self._propagate_bits(p1, l)
                    new_candidates.append(p1)

            new_candidates.sort(key=lambda p: p.pm)
            paths = new_candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
