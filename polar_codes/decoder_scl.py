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
    _g_boxplus,
    _update_bits,
    _update_llrs,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_division(bits, poly, crc_length):
    reg = [0] * crc_length
    for bit in bits:
        feedback = bit ^ reg[0]
        reg = reg[1:] + [0]
        if feedback:
            for i in range(crc_length):
                if (poly >> (crc_length - 1 - i)) & 1:
                    reg[i] ^= feedback
    return np.array(reg, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    if crc_length not in CRC_POLYNOMIALS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_division(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length not in CRC_POLYNOMIALS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_division(bits[:-crc_length], poly, crc_length)
    return np.array_equal(remainder, bits[-crc_length:])


class _PathState:
    __slots__ = ("pm", "B", "u_hat", "llr_idx")

    def __init__(self, N, n):
        self.pm = 0.0
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)
        self.llr_idx = 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.brev = bit_reversal_permutation(N)
        self.info_positions = np.where(self.frozen_bits == 0)[0]

        self.llr_pool = [
            np.zeros((N, self.n + 1), dtype=np.float64) for _ in range(list_size)
        ]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.brev].copy()

        paths = [_PathState(self.N, self.n) for _ in range(self.list_size)]
        for idx, llr_arr in enumerate(self.llr_pool):
            llr_arr.fill(0.0)
            llr_arr[:, 0] = llr
            paths[idx].llr_idx = idx

        active = [0]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []

            for pidx in active:
                path = paths[pidx]
                L = self.llr_pool[path.llr_idx]
                _update_llrs(L, path.B, l, self.n)
                llr_bit = L[l, self.n]

                if self.frozen_bits[l]:
                    candidates.append((path.pm + self._path_metric_penalty(llr_bit, 0), pidx, 0))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (path.pm + self._path_metric_penalty(llr_bit, bit), pidx, bit)
                        )

            candidates.sort(key=lambda item: item[0])
            survivors = candidates[: self.list_size]

            new_active = []
            used_llr = set()
            for slot, (pm, src_idx, bit) in enumerate(survivors):
                dst_idx = slot
                src = paths[src_idx]
                dst = paths[dst_idx]

                if dst_idx != src_idx:
                    dst.pm = pm
                    dst.u_hat = src.u_hat.copy()
                    dst.B = src.B.copy()
                    if src.llr_idx not in used_llr:
                        dst.llr_idx = src.llr_idx
                    else:
                        free_idx = next(
                            i for i in range(self.list_size) if i not in used_llr
                        )
                        self.llr_pool[free_idx][:] = self.llr_pool[src.llr_idx]
                        dst.llr_idx = free_idx
                else:
                    dst.pm = pm

                dst.u_hat[l] = bit
                dst.B[l, self.n] = bit
                L = self.llr_pool[dst.llr_idx]
                _update_bits(dst.B, l, self.n)
                new_active.append(dst_idx)
                used_llr.add(dst.llr_idx)

            active = new_active

        best_idx = active[0]
        best_pm = paths[best_idx].pm
        crc_best = None

        if self.crc_length > 0:
            for pidx in active:
                path = paths[pidx]
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    if crc_best is None or path.pm < paths[crc_best].pm:
                        crc_best = pidx

        if crc_best is not None:
            best_idx = crc_best
            best_pm = paths[best_idx].pm
        else:
            for pidx in active:
                if paths[pidx].pm < best_pm:
                    best_pm = paths[pidx].pm
                    best_idx = pidx

        return paths[best_idx].u_hat.copy(), best_pm
