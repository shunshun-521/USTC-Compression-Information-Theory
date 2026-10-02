"""
极化码 SCL（串行抵消列表）译码器，含 CRC 辅助 CA-SCL
"""
import numpy as np
from decoder_sc import (
    _SCDState,
    bit_reversed,
    active_llr_level,
    active_bit_level,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("state", "pm", "u_hat")

    def __init__(self, N, llr_ch):
        self.state = _SCDState(N, llr_ch.copy())
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def clone(self):
        p = _Path(self.state.N, np.zeros(self.state.N))
        p.state.L = self.state.L.copy()
        p.state.B = self.state.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_idx = np.sort(np.where(~self.frozen_bits)[0])

    def _pm_update(self, pm, llr_val, u_bit):
        v = 0 if llr_val >= 0 else 1
        if u_bit != v:
            pm += abs(llr_val)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1:
            from decoder_sc import sc_decode

            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0
        paths = [_Path(self.N, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for p in paths:
                p.state.update_llrs(l)
                llr_bit = p.state.L[l, self.n]
                if self.frozen_bits[i]:
                    cp = p.clone()
                    cp.pm = self._pm_update(cp.pm, llr_bit, 0)
                    cp.state.B[l, self.n] = 0
                    cp.state.update_bits(l)
                    cp.u_hat[i] = 0
                    new_paths.append(cp)
                else:
                    for bit in (0, 1):
                        cp = p.clone()
                        cp.pm = self._pm_update(cp.pm, llr_bit, bit)
                        cp.state.B[l, self.n] = bit
                        cp.state.update_bits(l)
                        cp.u_hat[i] = bit
                        new_paths.append(cp)
            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.L]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_idx], self.crc_length)]
            if valid:
                paths = valid
        best = min(paths, key=lambda x: x.pm)
        return best.u_hat, best.pm
