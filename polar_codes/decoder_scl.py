"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length=8):
    poly = CRC_POLYS[crc_length]
    n = crc_length
    reg = [int(b) for b in bits] + [0] * n
    for i in range(len(bits)):
        if reg[i]:
            for j in range(n + 1):
                if (poly >> j) & 1:
                    reg[i + j] ^= 1
    return np.array(reg[-n:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return np.all(_crc_remainder(bits, crc_length) == 0)


def _pm_update(pm, llr_val, u):
    hard = 0 if llr_val >= 0 else 1
    if u != hard:
        pm += abs(llr_val)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [int(bit_reversal_permutation(N)[i]) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    ) & 1
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数。"""
        brp = bit_reversal_permutation(self.N)
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[brp]
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = self.decode_order[phi]
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_dec = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm = _pm_update(path.pm, llr_dec, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = _Path(self.N, self.n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = _pm_update(path.pm, llr_dec, u)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
