"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _update_llr,
    _update_bits,
    sc_decode,
)
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
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
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.uint8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 L/B 数组，分裂时复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L = list_size
        self.crc_length = crc_length
        if list_size == 1 and crc_length == 0:
            self._sc_only = True
        else:
            self._sc_only = False

    def _llr_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self._sc_only:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        rev = bit_reversal_permutation(self.N)
        llr = llr_ch[rev]

        paths = [_Path(self.N, self.n) for _ in range(self.L)]
        for p in paths:
            p.L[:, self.n] = llr

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []
            for p in paths:
                if not p.active:
                    continue
                _update_llr(p.L, p.B, l, self.n, self.N)
                cur_llr = p.L[l, 0]
                if l in self.frozen_set:
                    bit = 0
                    pm = p.pm + self._llr_penalty(cur_llr, 0)
                    np_p = self._fork_path(p)
                    np_p.pm = pm
                    np_p.u_hat[l] = 0
                    np_p.B[l, 0] = 0
                    _update_bits(np_p.B, l, self.n, self.N)
                    new_paths.append(np_p)
                else:
                    for bit in (0, 1):
                        pm = p.pm + self._llr_penalty(cur_llr, bit)
                        np_p = self._fork_path(p)
                        np_p.pm = pm
                        np_p.u_hat[l] = bit
                        np_p.B[l, 0] = bit
                        _update_bits(np_p.B, l, self.n, self.N)
                        new_paths.append(np_p)

            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.L]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            payload_len = len(info_idx) - self.crc_length
            valid = []
            for p in paths:
                bits = p.u_hat[info_idx]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda x: x.pm)

        return best.u_hat.astype(int), float(best.pm)

    def _fork_path(self, p):
        q = _Path(self.N, self.n)
        q.L = p.L.copy()
        q.B = p.B.copy()
        q.pm = p.pm
        q.u_hat = p.u_hat.copy()
        return q
