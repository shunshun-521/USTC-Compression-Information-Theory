"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, width):
    reg ^= int(bit) << (width - 1)
    for _ in range(8):
        if reg & (1 << (width - 1)):
            reg = ((reg << 1) & ((1 << width) - 1)) ^ poly
        else:
            reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg = _crc_step(reg, bit, poly, crc_length)
    return reg == 0


class PathState:
    """单条路径状态。"""

    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src, dst):
        dst.pm = src.pm
        dst.u_hat[:] = src.u_hat
        dst.L[:] = src.L
        dst.B[:] = src.B

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top = path.L[j, s]
                    btm = path.L[j + branch_size, s]
                    if np.isnan(top):
                        top = 0.0
                    if np.isnan(btm):
                        btm = 0.0
                    path.L[j, s + 1] = f_operation(top, btm)
                else:
                    top = path.L[j - branch_size, s]
                    btm = path.L[j, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    if np.isnan(top):
                        top = 0.0
                    if np.isnan(btm):
                        btm = 0.0
                    path.L[j, s + 1] = g_operation(top, btm, int(top_bit))

    def _update_bits(self, path, l, u_val):
        path.u_hat[l] = u_val
        path.B[l, self.n] = u_val
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    b_j = int(path.B[j, s])
                    b_top = 0 if np.isnan(path.B[j - branch_size, s]) else int(path.B[j - branch_size, s])
                    path.B[j - branch_size, s - 1] = b_j ^ b_top
                    path.B[j, s - 1] = b_j

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch
        pool = [PathState(self.N, self.n) for _ in range(self.list_size - 1)]

        for phi in range(self.N):
            l = bit_reversed_index(phi, self.n)
            candidates = []

            for p_idx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if self.frozen_bits[l]:
                    penalty = self._pm_penalty(llr, 0)
                    path.pm += penalty
                    self._update_bits(path, l, 0)
                    candidates.append((path.pm, p_idx, None))
                else:
                    for u in (0, 1):
                        candidates.append((path.pm + self._pm_penalty(llr, u), p_idx, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            used_src = set()
            for pm, src_idx, u_val in candidates:
                if u_val is None:
                    paths[src_idx].pm = pm
                    new_paths.append(paths[src_idx])
                    used_src.add(src_idx)
                else:
                    if src_idx not in used_src:
                        dst = paths[src_idx]
                        used_src.add(src_idx)
                    elif pool:
                        dst = pool.pop()
                        self._copy_path(paths[src_idx], dst)
                    else:
                        dst = PathState(self.N, self.n)
                        self._copy_path(paths[src_idx], dst)
                    dst.pm = pm
                    self._update_bits(dst, l, u_val)
                    new_paths.append(dst)

            paths = new_paths

        crc_paths = []
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)

        search = crc_paths if crc_paths else paths
        best = min(search, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
