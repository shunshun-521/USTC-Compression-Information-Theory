"""
极化码 SCL（串行抵消列表）译码器，含 CRC 辅助 CA-SCL
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in np.asarray(bits, dtype=int):
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    reg = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    return _crc_remainder(bits, crc_length) == 0


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _log_sum(a, b):
    if a >= b:
        return a + np.log1p(np.exp(b - a))
    return b + np.log1p(np.exp(a - b))


def _upper_llr(l1, l2):
    return _log_sum(l1 + l2, 0.0) - _log_sum(l1, l2)


def _lower_llr(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2


class _SCState:
    """单路径 SC 状态（Permuted SCD）"""

    def __init__(self, llr_tree, n, N):
        self.n = n
        self.N = N
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_tree
        self.pm = 0.0
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + half, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j - half, s], self.L[j, s], self.B[j - half, s + 1]
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 2 ** s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    self.B[j - half, s - 1] = int(self.B[j, s]) ^ int(self.B[j - half, s])
                    self.B[j, s - 1] = self.B[j, s]

    def copy(self):
        st = _SCState(self.L[:, 0].copy(), self.n, self.N)
        st.L = self.L.copy()
        st.B = self.B.copy()
        st.pm = self.pm
        return st

    def u_hat(self):
        return self.B[:, self.n].astype(int)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = int(list_size)
        self.crc_length = int(crc_length)
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_tree = llr_ch[self.br]
        paths = [_SCState(llr_tree, self.n, self.N)]

        for i in range(self.N):
            l = self.br[i]
            candidates = []
            for p in paths:
                p._update_llrs(l)
                llr = float(p.L[l, self.n])
                if self.frozen_bits[l]:
                    p.B[l, self.n] = 0
                    if llr < 0:
                        p.pm += abs(llr)
                    p._update_bits(l)
                    candidates.append(p)
                else:
                    for bit in (0, 1):
                        cp = p.copy()
                        cp.B[l, self.n] = bit
                        if (bit == 0 and llr < 0) or (bit == 1 and llr >= 0):
                            cp.pm += abs(llr)
                        cp._update_bits(l)
                        candidates.append(cp)
            candidates.sort(key=lambda x: x.pm)
            paths = candidates[: self.L]

        best = min(paths, key=lambda x: x.pm)
        u_hat = best.u_hat()

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            K_info = len(info_idx) - self.crc_length
            valid = []
            for p in paths:
                bits = p.u_hat()[info_idx]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda x: x.pm)
                u_hat = best.u_hat()

        return u_hat, best.pm
