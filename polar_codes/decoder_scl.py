"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _lower_llr,
    _prepare_channel_llr,
    f_operation,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, crc_length, poly):
    """LFSR CRC 处理，返回寄存器状态"""
    state = np.zeros(crc_length, dtype=int)
    poly_bits = np.array(
        [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    for bit in bits:
        fb = state[0] ^ int(bit)
        state[:-1] = state[1:]
        state[-1] = 0
        if fb:
            state ^= poly_bits
    return state


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc_bits = _crc_process(info_bits, crc_length, poly)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    state = _crc_process(bits, crc_length, poly)
    return np.all(state == 0)


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src):
        dst = _PathState(self.N, self.n)
        dst.L = src.L
        dst.B = src.B
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        return dst

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_tree = _prepare_channel_llr(llr_ch)
        paths = [_PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_tree

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    candidates.append((_pm_update(path.pm, llr, 0), pidx, 0))
                else:
                    for u in (0, 1):
                        candidates.append((_pm_update(path.pm, llr, u), pidx, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for new_pm, pidx, u in candidates:
                np_ = self._copy_path(paths[pidx])
                np_.pm = new_pm
                np_.u_hat[l] = u
                np_.B[l, self.n] = u
                self._update_bits(np_, l)
                new_paths.append(np_)

            paths = new_paths

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
        return best.u_hat.astype(int), best.pm
