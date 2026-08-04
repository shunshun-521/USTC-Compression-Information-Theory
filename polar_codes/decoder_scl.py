"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed,
    _active_llr_level, _active_bit_level,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ bit) & 1
        reg = ((reg << 1) & mask)
        if fb:
            reg ^= poly

    crc_bits = np.zeros(crc_length, dtype=np.int64)
    for i in range(crc_length):
        crc_bits[i] = (reg >> (crc_length - 1 - i)) & 1

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded[-crc_length:], bits[-crc_length:])


class _Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ('L', 'B', 'pm', 'u_hat', '_owns_L', '_owns_B')

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int64)
        self._owns_L = True
        self._owns_B = True

    def fork(self):
        child = _Path.__new__(_Path)
        child.L = self.L
        child.B = self.B
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        child._owns_L = False
        child._owns_B = False
        return child

    def ensure_L(self):
        if not self._owns_L:
            self.L = self.L.copy()
            self._owns_L = True

    def ensure_B(self):
        if not self._owns_B:
            self.B = self.B.copy()
            self._owns_B = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

        rev = np.zeros(N, dtype=np.int64)
        for i in range(N):
            r = 0
            v = i
            for _ in range(self.n):
                r = (r << 1) | (v & 1)
                v >>= 1
            rev[i] = r
        self._rev = rev

    def _prepare_llr(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64).copy()
        return llr[self._rev]

    def _update_llrs(self, path, l):
        path.ensure_L()
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        path.ensure_B()
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = self._prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phase in range(self.N):
            l = _bit_reversed(phase, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    path.pm += penalty
                    path.u_hat[l] = 0
                    path.ensure_B()
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = path.fork()
                        if bit == 0:
                            penalty = abs(llr_val) if llr_val < 0 else 0.0
                        else:
                            penalty = abs(llr_val) if llr_val >= 0 else 0.0
                        child.pm += penalty
                        child.u_hat[l] = bit
                        child.ensure_B()
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)

        return best.u_hat, best.pm
