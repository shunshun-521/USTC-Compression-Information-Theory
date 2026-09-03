"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    _lower_llr,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length, poly):
    """计算 CRC 余数。"""
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    rem = _crc_remainder(info_bits, crc_length, poly)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    data = bits[:-crc_length]
    expected = crc_encode(data, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _path_metric_update(self, pm, llr, u_bit):
        preferred = 0 if llr >= 0 else 1
        if u_bit != preferred:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr0 = llr_ch[self.rev]
        N, n = self.N, self.n

        class Path:
            def __init__(self):
                self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
                self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
                self.L[:, 0] = llr0
                self.pm = 0.0

        paths = [Path()]
        decode_order = [_bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            is_frozen = l in self.frozen_set
            new_paths = []

            for path in paths:
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            path.L[j, s + 1] = f_operation(
                                path.L[j, s], path.L[j + branch_size, s]
                            )
                        else:
                            top_bit = path.B[j - branch_size, s + 1]
                            path.L[j, s + 1] = _lower_llr(
                                path.L[j, s],
                                path.L[j - branch_size, s],
                                path.B[j - branch_size, s + 1],
                            )
                llr = path.L[l, n]

                if is_frozen:
                    u_bit = 0
                    path.pm = self._path_metric_update(path.pm, llr, u_bit)
                    path.B[l, n] = 0
                    self._update_bits(path, l, n)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p = Path()
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.pm = self._path_metric_update(path.pm, llr, u_bit)
                        p.B[l, n] = u_bit
                        self._update_bits(p, l, n)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        crc_pass = []
        for p in paths:
            u_hat = p.B[:, n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        if crc_pass:
            best = min(crc_pass, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, n].astype(int), best.pm

    def _update_bits(self, path, l, n):
        if l < self.N / 2:
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
