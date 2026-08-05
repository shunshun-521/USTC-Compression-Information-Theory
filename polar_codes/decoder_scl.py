"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation, g_operation,
    _active_llr_level, _active_bit_level,
)


CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]
CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1]


def _get_gen(crc_length):
    return CRC8_GEN if crc_length == 8 else CRC16_GEN


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    gen = _get_gen(crc_length)
    padded = list(info_bits) + [0] * crc_length
    for i in range(len(info_bits)):
        if padded[i]:
            for j in range(len(gen)):
                if i + j < len(padded):
                    padded[i + j] ^= gen[j]
    crc_bits = np.array(padded[-crc_length:], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    gen = _get_gen(crc_length)
    padded = list(bits)
    for i in range(len(bits) - crc_length):
        if padded[i]:
            for j in range(len(gen)):
                if i + j < len(padded):
                    padded[i + j] ^= gen[j]
    return all(x == 0 for x in padded[-crc_length:])


class Path:
    """SCL 译码单条路径"""

    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src):
        p = Path(self.N, self.n)
        p.pm = src.pm
        p.L = src.L
        p.B = src.B.copy()
        p.u_hat = src.u_hat.copy()
        return p

    def _update_llrs(self, path, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] + path.B[j - branch_size, s]
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        paths = [Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(N):
            l = bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if self.frozen_bits[l]:
                    new_p = self._copy_path(path)
                    new_p.pm += self._pm_penalty(llr, 0)
                    new_p.B[l, n] = 0
                    new_p.u_hat[l] = 0
                    self._update_bits(new_p, l)
                    new_paths.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = self._copy_path(path)
                        new_p.pm += self._pm_penalty(llr, u)
                        new_p.B[l, n] = u
                        new_p.u_hat[l] = u
                        self._update_bits(new_p, l)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                paths = crc_pass

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
