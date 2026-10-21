"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
    _update_llrs,
    _update_bits,
)


def _crc_generator(crc_length):
    if crc_length == 8:
        return 0x107  # x^8 + x^2 + x + 1 (CRC-8, poly 0x07)
    if crc_length == 16:
        return 0x18005  # CRC-16 (poly 0x8005)
    raise ValueError("crc_length must be 8 or 16")


def _gf2_crc_remainder(msg_bits, crc_length, generator):
    gen_bits = [(generator >> i) & 1 for i in range(crc_length, -1, -1)]
    n = len(msg_bits)
    v = list(map(int, msg_bits)) + [0] * crc_length
    for i in range(n):
        if v[i]:
            for j in range(len(gen_bits)):
                v[i + j] ^= gen_bits[j]
    return v[n : n + crc_length]


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    generator = _crc_generator(crc_length)
    rem = _gf2_crc_remainder(info_bits, crc_length, generator)
    crc_bits = np.array(rem, dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    generator = _crc_generator(crc_length)
    rem = _gf2_crc_remainder(bits[:-crc_length], crc_length, generator)
    return np.array_equal(rem, bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_paths(self, llr_ch):
        p = _Path(self.N, self.n)
        p.L[:, 0] = llr_ch
        return [p]

    def _path_llr_at(self, path, i):
        l = bit_reversed(i, self.n)
        _update_llrs(path.L, path.B, l, self.n, self.N)
        return path.L[l, self.n]

    def _path_propagate_bit(self, path, i, bit):
        l = bit_reversed(i, self.n)
        path.u_hat[i] = bit
        path.B[l, self.n] = bit
        _update_bits(path.B, l, self.n, self.N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for i in range(self.N):
            candidates = []
            for path in paths:
                if not path.active:
                    continue
                llr_i = self._path_llr_at(path, i)

                if self.frozen_bits[i]:
                    bit = 0
                    penalty = 0.0 if llr_i >= 0 else abs(llr_i)
                    new_pm = path.pm + penalty
                    candidates.append((new_pm, path, bit))
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_i >= 0) or (bit == 1 and llr_i < 0) else abs(llr_i)
                        candidates.append((path.pm + penalty, path, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in selected:
                child = _Path(self.N, self.n)
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.pm = pm
                child.u_hat = parent.u_hat.copy()
                self._path_propagate_bit(child, i, bit)
                new_paths.append(child)
            paths = new_paths

        # 选择输出路径
        best = None
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            info_bits_all = np.where(info_mask)[0]
            for p in paths:
                bits = p.u_hat[info_bits_all]
                if crc_check(bits, self.crc_length):
                    if best is None or p.pm < best.pm:
                        best = p
        if best is None:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
