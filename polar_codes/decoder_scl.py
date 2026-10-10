"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_logdomain, g_logdomain, _bit_reversed_index


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_update(reg, bit, crc_length, poly):
    msb = 1 << (crc_length - 1)
    reg ^= int(bit) << (crc_length - 1)
    if reg & msb:
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位（MSB 优先），附加在信息比特后。"""
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg = _crc_update(reg, b, crc_length, poly)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, crc_bits]).astype(np.int32)


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg = _crc_update(reg, b, crc_length, poly)
    return reg == 0


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        brp = bit_reversal_permutation(N)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, n] = llr_ch[brp]
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)

    def copy(self):
        p = _PathState.__new__(_PathState)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


def _update_llrs_for_bit(L, B, i, n, N):
    l = _bit_reversed_index(i, n)
    for j in range(n - 1, -1, -1):
        s = 2 ** (n - j)
        t = s // 2
        for idx in range(l, N, s):
            if t > idx % s:
                L[idx, j] = f_logdomain(L[idx, j + 1], L[idx + t, j + 1])
            else:
                L[idx, j] = g_logdomain(L[idx, j + 1], L[idx - t, j + 1], int(B[idx - t, j]))
    return L[l, 0]


def _propagate_bit(L, B, i, n, N, bit):
    l = _bit_reversed_index(i, n)
    B[l, 0] = bit
    active = [l]
    for j in range(n):
        s = 2 ** (n - j)
        t = s // 2
        nxt = []
        for idx in active:
            if t <= idx % s:
                B[idx - t, j + 1] = (B[idx, j] + B[idx - t, j]) % 2
                B[idx, j + 1] = B[idx, j]
                nxt.extend([idx, idx - t])
        active = nxt


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=np.int64)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )

    @staticmethod
    def _pm_add(llr, u_bit):
        v = 0 if llr >= 0 else 1
        return 0.0 if u_bit == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []
            for path in paths:
                llr = _update_llrs_for_bit(path.L, path.B, i, self.n, self.N)
                if l in self.frozen_set:
                    child = path.copy()
                    child.pm += self._pm_add(llr, 0)
                    _propagate_bit(child.L, child.B, i, self.n, self.N, 0)
                    child.u_hat[l] = 0
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm += self._pm_add(llr, bit)
                        _propagate_bit(child.L, child.B, i, self.n, self.N, bit)
                        child.u_hat[l] = bit
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            ok = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if ok:
                best = min(ok, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
