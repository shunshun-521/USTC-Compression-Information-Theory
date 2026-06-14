"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _hard_decision,
    _lower_llr,
    _update_bits,
    _update_llrs,
    _upper_llr,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_update(pm, llr, u):
    hard = _hard_decision(llr)
    if u != hard:
        return pm + abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（置换 SC + Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _compute_llr(self, path, l):
        _update_llrs(path.L, path.B, l, self.n, self.N)
        return path.L[l, self.n]

    def _propagate_bits(self, path, l):
        _update_bits(path.B, l, self.n, self.N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_internal = llr_ch[br]

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_internal

        for l in self.decode_order:
            candidates = []
            for pidx, path in enumerate(paths):
                llr = self._compute_llr(path, l)
                if l in self.frozen_set:
                    pm = _path_metric_update(path.pm, llr, 0)
                    path.pm = pm
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._propagate_bits(path, l)
                    candidates.append((pm, pidx, None))
                else:
                    for u in (0, 1):
                        pm = _path_metric_update(path.pm, llr, u)
                        candidates.append((pm, pidx, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, u_choice in candidates:
                if u_choice is None:
                    new_path = paths[pidx]
                    new_path.pm = pm
                else:
                    src = paths[pidx]
                    new_path = _Path(self.N, self.n)
                    new_path.L[:] = src.L
                    new_path.B[:] = src.B
                    new_path.u_hat[:] = src.u_hat
                    new_path.pm = pm
                    new_path.u_hat[l] = u_choice
                    new_path.B[l, self.n] = u_choice
                    self._propagate_bits(new_path, l)
                new_paths.append(new_path)
            paths = new_paths

        if self.crc_length > 0:
            crc_pass = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append((path.pm, path))
            if crc_pass:
                crc_pass.sort(key=lambda x: x[0])
                best = crc_pass[0][1]
                return best.u_hat.copy(), crc_pass[0][0]

        paths.sort(key=lambda p: p.pm)
        return paths[0].u_hat.copy(), paths[0].pm
