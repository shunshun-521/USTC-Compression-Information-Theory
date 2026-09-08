"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _bit_reversed_index,
    _prepare_frozen_bits,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversal_permutation


CRC_POLYS = {
    8: np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8),
    16: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8),
}


def _crc_remainder(bits, poly):
    msg = np.asarray(bits, dtype=np.int8).copy()
    r = len(poly) - 1
    for i in range(len(msg) - r):
        if msg[i]:
            msg[i:i + len(poly)] ^= poly
    return msg[-r:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYS[crc_length]
    r = len(poly) - 1
    padded = np.concatenate([np.asarray(info_bits, dtype=np.int8), np.zeros(r, dtype=np.int8)])
    remainder = _crc_remainder(padded, poly)
    return np.concatenate([np.asarray(info_bits, dtype=np.int8), remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确的 CRC。"""
    poly = CRC_POLYS[crc_length]
    msg = np.asarray(bits, dtype=np.int8).copy()
    r = len(poly) - 1
    for i in range(len(msg) - r):
        if msg[i]:
            msg[i:i + len(poly)] ^= poly
    return np.all(msg[-r:] == 0)


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class Path:
    """单条 SCL 路径。"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_init):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_init
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _prepare_frozen_bits(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

        info_mask = ~self.frozen_bits
        self.info_indices = np.where(info_mask)[0]
        if crc_length > 0:
            self.crc_info_indices = self.info_indices[: len(self.info_indices) - crc_length]
        else:
            self.crc_info_indices = self.info_indices

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = [Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for p_idx, path in enumerate(paths):
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pm = _pm_update(path.pm, llr, 0)
                    candidates.append((pm, p_idx, 0))
                else:
                    candidates.append((_pm_update(path.pm, llr, 0), p_idx, 0))
                    candidates.append((_pm_update(path.pm, llr, 1), p_idx, 1))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent_idx, u_bit in candidates:
                parent = paths[parent_idx]
                child = Path(self.N, self.n, parent.L[:, 0].copy())
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.B[l, self.n] = u_bit
                child.u_hat[l] = u_bit
                _update_bits(child.B, l, self.n)
                new_paths.append(child)

            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for idx, path in enumerate(paths):
                info_bits = path.u_hat[self.crc_info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((path.pm, idx))
            if valid:
                best_idx = min(valid, key=lambda x: x[0])[1]
            else:
                best_idx = int(np.argmin([p.pm for p in paths]))
        else:
            best_idx = int(np.argmin([p.pm for p in paths]))

        best = paths[best_idx]
        return best.u_hat.astype(int), best.pm
