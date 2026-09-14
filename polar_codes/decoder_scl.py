"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed,
    _active_llr_level, _active_bit_level,
)


CRC_POLYS = {
    8: [1, 0, 0, 0, 0, 0, 1, 1, 1],
    16: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
}


def _crc_remainder(bits, crc_length):
    """多项式除法求 CRC 余数。"""
    poly = CRC_POLYS[crc_length]
    d = list(np.asarray(bits, dtype=int))
    r = crc_length
    for i in range(len(d) - r):
        if d[i] == 1:
            for j in range(len(poly)):
                d[i + j] ^= poly[j]
    return d[-r:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(msg, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return _crc_remainder(bits, crc_length) == [0] * crc_length


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat', 'copied')

    def __init__(self, N, n, parent=None):
        self.copied = parent is None
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.u_hat = np.zeros(N, dtype=int)
            self.pm = 0.0
        else:
            self.L = parent.L
            self.B = parent.B
            self.u_hat = parent.u_hat.copy()
            self.pm = parent.pm
            self.copied = False

    def ensure_copy(self):
        if not self.copied:
            self.L = self.L.copy()
            self.B = self.B.copy()
            self.copied = True


def _update_llrs_path(path, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
            else:
                top_bit = path.B[j - branch_size, s + 1]
                path.L[j, s + 1] = g_operation(
                    path.L[j - branch_size, s], path.L[j, s], top_bit
                )


def _update_bits_path(path, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                path.B[j, s - 1] = path.B[j, s]


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N, n = self.N, self.n
        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                path.ensure_copy()
                _update_llrs_path(path, l, n, N)
                llr_val = path.L[l, n]

                if i in self.frozen_set:
                    penalty = self._pm_penalty(llr_val, 0)
                    path.pm += penalty
                    path.B[l, n] = 0
                    path.u_hat[i] = 0
                    _update_bits_path(path, l, n, N)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(N, n, parent=path)
                        child.ensure_copy()
                        child.pm = path.pm + self._pm_penalty(llr_val, u_bit)
                        child.B[l, n] = u_bit
                        child.u_hat[i] = u_bit
                        _update_bits_path(child, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(crc_pass or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
