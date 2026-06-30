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


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    n_crc = crc_length

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (n_crc - 1)
        for _ in range(n_crc):
            if reg & (1 << (n_crc - 1)):
                reg = ((reg << 1) & ((1 << n_crc) - 1)) ^ poly
            else:
                reg = (reg << 1) & ((1 << n_crc) - 1)

    crc_bits = np.array([(reg >> (n_crc - 1 - i)) & 1 for i in range(n_crc)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, encoded)


class Path:
    """单条译码路径（Lazy Copy）。"""

    __slots__ = ('pm', 'L', 'B', 'u_hat', 'L_owner', 'B_owner')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int32)
        self.L_owner = True
        self.B_owner = True

    def copy(self):
        new_path = Path.__new__(Path)
        new_path.pm = self.pm
        new_path.L = self.L
        new_path.B = self.B
        new_path.u_hat = self.u_hat.copy()
        new_path.L_owner = False
        new_path.B_owner = False
        return new_path

    def ensure_L_copy(self):
        if not self.L_owner:
            self.L = self.L.copy()
            self.L_owner = True

    def ensure_B_copy(self):
        if not self.B_owner:
            self.B = self.B.copy()
            self.B_owner = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _pm_penalty(llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(cur_llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    path.ensure_B_copy()
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u_val in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(cur_llr, u_val)
                        new_path.u_hat[l] = u_val
                        new_path.ensure_L_copy()
                        new_path.ensure_B_copy()
                        new_path.B[l, self.n] = u_val
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            pool = crc_pass if crc_pass else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
