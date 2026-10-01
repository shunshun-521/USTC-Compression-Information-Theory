"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np

from decoder_sc import sc_decode, bit_reversed, upper_llr, lower_llr, active_llr_level, active_bit_level


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        fb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(b)) & mask
        if fb ^ int(b):
            reg ^= poly
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        fb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(b)) & mask
        if fb ^ int(b):
            reg ^= poly
    return reg == 0


class _PathState:
    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, state, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    state.L[j, s + 1] = upper_llr(state.L[j, s], state.L[j + branch_size, s])
                else:
                    state.L[j, s + 1] = lower_llr(
                        state.L[j - branch_size, s], state.L[j, s], state.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, state, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    state.B[j - branch_size, s - 1] = int(state.B[j, s]) ^ int(state.B[j - branch_size, s])
                    state.B[j, s - 1] = state.B[j, s]

    def _pm_add(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for state in paths:
                self._update_llrs(state, l)
                llr_bit = state.L[l, self.n]
                if l in self.frozen_set:
                    st = copy.deepcopy(state)
                    st.B[l, self.n] = 0
                    st.u_hat[l] = 0
                    st.pm = self._pm_add(st.pm, llr_bit, 0)
                    self._update_bits(st, l)
                    new_paths.append(st)
                else:
                    for bit in (0, 1):
                        st = copy.deepcopy(state)
                        st.B[l, self.n] = bit
                        st.u_hat[l] = bit
                        st.pm = self._pm_add(st.pm, llr_bit, bit)
                        self._update_bits(st, l)
                        new_paths.append(st)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return p.u_hat.copy(), p.pm
        best = paths[0]
        return best.u_hat.copy(), best.pm
