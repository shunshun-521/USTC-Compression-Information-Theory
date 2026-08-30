"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _frozen_mask_to_set,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & mask
        if reg & (1 << (crc_length - 1)):
            reg = (reg ^ poly) & mask

    crc_bits = np.zeros(crc_length, dtype=int)
    for i in range(crc_length):
        reg = (reg << 1) & mask
        if reg & (1 << (crc_length - 1)):
            reg = (reg ^ poly) & mask
        crc_bits[i] = (reg >> (crc_length - 1)) & 1

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length))


class PathState:
    """单条 SCL 路径状态（Lazy Copy）。"""

    __slots__ = ("pm", "L", "B", "u_hat", "parent", "copied")

    def __init__(self, N, n, llr_ch=None):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.u_hat = np.zeros(N, dtype=int)
        self.parent = None
        self.copied = True
        if llr_ch is not None:
            self.L[:, 0] = llr_ch

    def ensure_owned(self):
        if not self.copied:
            parent = self.parent
            self.L = parent.L.copy()
            self.B = parent.B.copy()
            self.u_hat = parent.u_hat.copy()
            self.pm = parent.pm
            self.copied = True
            self.parent = None

    def fork(self):
        child = PathState.__new__(PathState)
        child.pm = self.pm
        child.L = self.L
        child.B = self.B
        child.u_hat = self.u_hat
        child.parent = self
        child.copied = False
        return child


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_mask_to_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            sorted(set(range(N)) - self.frozen_set), dtype=int
        )

    def _update_llrs(self, path, l):
        path.ensure_owned()
        L, B = path.L, path.B
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _propagate_bits(self, path, l, bit):
        path.ensure_owned()
        path.u_hat[l] = bit
        path.B[l, self.n] = bit
        if l < self.N // 2:
            return
        B = path.B
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    @staticmethod
    def _path_metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = path.fork()
                    new_path.ensure_owned()
                    new_path.pm += self._path_metric_penalty(llr, 0)
                    self._propagate_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.fork()
                        new_path.ensure_owned()
                        new_path.pm += self._path_metric_penalty(llr, bit)
                        self._propagate_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
