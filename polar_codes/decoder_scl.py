"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed, _active_llr_level, _active_bit_level,
    _upper_llr, _lower_llr, f_operation, g_operation
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg == 0


class PathState:
    """单条译码路径的状态"""

    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def copy(self):
        p = PathState(self.N, self.n, self.L[:, 0])
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = _lower_llr(self.L[j, s], self.L[j - branch_size, s], top_bit)

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = self.B[j, s] ^ self.B[j - branch_size, s]
                    self.B[j, s - 1] = self.B[j, s]

    def get_llr(self, l):
        return self.L[l, self.n]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        N = self.N
        n = self.n
        paths = [PathState(N, n, llr_ch)]

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                path.update_llrs(l)
                llr_phi = path.get_llr(l)

                if self.frozen_bits[i]:
                    new_path = path.copy()
                    new_path.B[l, n] = 0
                    new_path.pm += self._pm_penalty(llr_phi, 0)
                    new_path.update_bits(l)
                    new_paths.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.B[l, n] = u
                        new_path.pm += self._pm_penalty(llr_phi, u)
                        new_path.update_bits(l)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            crc_pass = []
            for path in paths:
                u_hat = path.B[:, n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else paths[0]
        else:
            best = paths[0]

        rev_perm = np.array([bit_reversed(i, self.n) for i in range(self.N)])
        return best.B[rev_perm, self.n].astype(int), best.pm
