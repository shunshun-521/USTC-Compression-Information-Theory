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
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _llr_penalty(self, llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [_PathState(N, n) for _ in range(self.list_size)]
        paths[0].L[:, 0] = llr_ch
        active = 1

        for phase in range(N):
            l = _bit_reversed_index(phase, n)
            candidates = []

            for p_idx in range(active):
                path = paths[p_idx]
                _update_llrs(path.L, path.B, l, n)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    penalty = self._llr_penalty(llr, 0)
                    candidates.append((path.pm + penalty, p_idx, 0, True))
                else:
                    for u in (0, 1):
                        penalty = self._llr_penalty(llr, u)
                        candidates.append((path.pm + penalty, p_idx, u, False))

            candidates.sort(key=lambda x: x[0])
            new_paths = [_PathState(N, n) for _ in range(self.list_size)]
            new_active = 0

            for pm, parent_idx, u_choice, is_frozen in candidates:
                if new_active >= self.list_size:
                    break
                parent = paths[parent_idx]
                child = new_paths[new_active]
                child.L[:] = parent.L
                child.B[:] = parent.B
                child.pm = pm
                child.u_hat[:] = parent.u_hat

                bit = 0 if is_frozen else u_choice
                child.B[l, n] = bit
                child.u_hat[l] = bit
                _update_bits(child.B, l, n, N)
                new_active += 1

            paths = new_paths
            active = new_active

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                if crc_check(path.u_hat, self.crc_length):
                    return path.u_hat.copy(), path.pm

        best = paths[0]
        return best.u_hat.copy(), best.pm
