"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _b_check,
    _compute_llr,
    _update_partial_sums,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, poly, crc_len):
    state = 0
    mask = (1 << crc_len) - 1
    for b in bits:
        fb = ((state >> (crc_len - 1)) & 1) ^ int(b)
        state = (state << 1) & mask
        if fb:
            state ^= poly
    return state


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    poly = _crc_poly(crc_length)
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("llrs", "s", "pm", "u")

    def __init__(self, n, N):
        self.llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
        self.s = -np.ones((n + 1, N), dtype=np.int8)
        self.pm = 0.0
        self.u = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(int)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [_Path(n, N) for _ in range(self.list_size)]
        paths[0].llrs[n, :] = llr_ch
        active = 1

        for i in range(N):
            candidates = []
            for p_idx in range(active):
                path = paths[p_idx]
                if self.frozen_bits[i] == 1:
                    llr_i = _compute_llr(0, i, path.llrs, path.s)
                    pm = path.pm + self._path_metric_penalty(llr_i, 0)
                    candidates.append((pm, p_idx, 0, llr_i))
                else:
                    llr_i = _compute_llr(0, i, path.llrs, path.s)
                    for bit in (0, 1):
                        pm = path.pm + self._path_metric_penalty(llr_i, bit)
                        candidates.append((pm, p_idx, bit, llr_i))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent_idx, bit, llr_i in selected:
                parent = paths[parent_idx]
                child = _Path(n, N)
                child.llrs = parent.llrs.copy()
                child.s = parent.s.copy()
                child.u = parent.u.copy()
                child.pm = pm
                child.s[0, i] = bit
                child.u[i] = bit
                if self.frozen_bits[i] == 1:
                    child.s[0, i] = 0
                    child.u[i] = 0
                new_paths.append(child)
            paths = new_paths
            active = len(paths)

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            crc_ok = [p for p in paths if crc_check(p.u[info_idx], self.crc_length)]
            if crc_ok:
                best = min(crc_ok, key=lambda p: p.pm)
        return best.u.astype(int), best.pm
