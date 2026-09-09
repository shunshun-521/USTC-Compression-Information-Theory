"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _g_llr,
    _prepare_llr,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            poly_bits = np.array(
                [(poly >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
                dtype=np.int8,
            )
            reg ^= poly_bits
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Permuted SC + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._phase_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = _g_llr(path.L[j, s], path.L[j - branch_size, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for l in self._phase_order:
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr_val, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    candidates.append((path.pm, pidx, None))
                else:
                    for u in (0, 1):
                        candidates.append((path.pm + self._pm_penalty(llr_val, u), pidx, u))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            used = set()

            for pm, pidx, u in candidates:
                if len(new_paths) >= self.list_size:
                    break
                if self.frozen_bits[l]:
                    if pidx in used:
                        continue
                    used.add(pidx)
                    paths[pidx].pm = pm
                    new_paths.append(paths[pidx])
                else:
                    if pidx in used:
                        new_path = _Path(self.N, self.n)
                        src = paths[pidx]
                        new_path.pm = pm
                        new_path.L[:] = src.L
                        new_path.B[:] = src.B
                        new_path.u_hat[:] = src.u_hat
                        new_path.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        new_paths.append(new_path)
                    else:
                        used.add(pidx)
                        paths[pidx].pm = pm
                        paths[pidx].B[l, self.n] = u
                        paths[pidx].u_hat[l] = u
                        self._update_bits(paths[pidx], l)
                        new_paths.append(paths[pidx])

            paths = new_paths

        pool = paths
        if self.crc_length > 0:
            passed = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(path)
            if passed:
                pool = passed

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
