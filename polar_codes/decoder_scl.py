"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, permute_llr_for_decode,
    _bit_reversed_index, _active_llr_level, _active_bit_level,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    poly = CRC_POLYNOMIALS[crc_length]
    bits = np.asarray(bits, dtype=int)
    reg = 0
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    return reg == 0


class PathState:
    """单条 SCL 路径状态"""
    __slots__ = ('L', 'B', 'pm', 'u')

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)

    def copy(self):
        new = PathState.__new__(PathState)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u = self.u.copy()
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    btm = path.L[j, s]
                    top = path.L[j - branch_size, s]
                    bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(top, btm, bit)

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        N, n = self.N, self.n
        llr = permute_llr_for_decode(llr_ch).astype(np.float64)
        paths = [PathState(N, n, llr)]

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm = self._pm_update(path.pm, llr_val, 0)
                    new_path.u[l] = 0
                    new_path.B[l, n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = path.copy()
                        new_path.pm = self._pm_update(path.pm, llr_val, u_val)
                        new_path.u[l] = u_val
                        new_path.B[l, n] = u_val
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            crc_ok = []
            for p in paths:
                info_bits = p.u[self.frozen_bits == 0]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(p)
            if crc_ok:
                paths = crc_ok

        best = min(paths, key=lambda p: p.pm)
        return best.u, best.pm
