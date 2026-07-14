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


CRC8_POLY = 0x07
CRC16_POLY = 0x8005

_CRC_POLY_BITS = {
    8: [8, 2, 1, 0],
    16: [16, 15, 2, 0],
}


def _poly_from_bits(loc, crc_n):
    p = [0] * (crc_n + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def _crc_division(info_bits, crc_length):
    """GF(2) 多项式长除法求 CRC 余数。"""
    loc = _CRC_POLY_BITS[crc_length]
    p = _poly_from_bits(loc, crc_length)
    info = [int(b) for b in info_bits]
    times = len(info)
    for _ in range(crc_length):
        info.append(0)
    for i in range(times):
        if info[i] == 1:
            for j in range(crc_length + 1):
                info[j + i] ^= p[j]
    return np.array(info[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    check_code = _crc_division(info_bits, crc_length)
    return np.concatenate([info_bits, check_code])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """
    SCL 译码器（路径复制实现）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, path, l):
        start = self.n - _active_llr_level(l, self.n)
        for s in range(start, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
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
        abl = _active_bit_level(l, self.n)
        for s in range(self.n, self.n - abl, -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = int(f"{i:0{self.n}b}"[::-1], 2)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[i]:
                    bit = 0
                    pm = path.pm + self._pm_penalty(llr, bit)
                    candidates.append((pm, path, bit))
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._pm_penalty(llr, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in selected:
                child = _Path(self.N, self.n)
                child.L[:] = parent.L
                child.B[:] = parent.B
                child.pm = pm
                child.u_hat[:] = parent.u_hat
                child.u_hat[i] = bit
                child.B[l, self.n] = bit
                self._update_bits(child, l)
                new_paths.append(child)

            paths = new_paths

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
