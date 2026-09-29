"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversal_index,
    _update_llr,
    _update_bits,
    _f_boxplus,
    _g_boxplus,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加到信息比特后"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8).ravel()
    for b in bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, n] = llr_ch.copy()
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = max(1, int(list_size))
        self.crc_length = int(crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phase in range(self.N):
            l = bit_reversal_index(phase, self.n)
            candidates = []

            for path in paths:
                _update_llr(path.L, path.B, l, self.n, self.N)
                llr_bit = path.L[l, 0]
                if self.frozen_bits[l]:
                    pen = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    new_pm = path.pm + pen
                    p_new = _Path(self.N, self.n, llr_ch)
                    p_new.L = path.L.copy()
                    p_new.B = path.B.copy()
                    p_new.pm = new_pm
                    p_new.B[l, 0] = 0
                    _update_bits(p_new.B, l, self.n, self.N)
                    candidates.append(p_new)
                else:
                    for u_bit in (0, 1):
                        pen = 0.0 if (u_bit == 0 and llr_bit >= 0) or (u_bit == 1 and llr_bit < 0) else abs(llr_bit)
                        p_new = _Path(self.N, self.n, llr_ch)
                        p_new.L = path.L.copy()
                        p_new.B = path.B.copy()
                        p_new.pm = path.pm + pen
                        p_new.B[l, 0] = u_bit
                        _update_bits(p_new.B, l, self.n, self.N)
                        candidates.append(p_new)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.L]

        best_crc = None
        best_pm = None
        for p in paths:
            u_hat = p.B[:, 0].astype(int)
            pm = p.pm
            if self.crc_length > 0:
                info_idx = np.where(~self.frozen_bits)[0]
                payload = u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or pm < best_crc.pm:
                        best_crc = p
            if best_pm is None or pm < best_pm:
                best_pm = p

        chosen = best_crc if best_crc is not None else best_pm
        return chosen.B[:, 0].astype(int), chosen.pm
