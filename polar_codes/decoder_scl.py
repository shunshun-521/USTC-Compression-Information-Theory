"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr_exact,
    _lower_llr,
    sc_decode,
)
CRC8_POLY = 0x07
CRC16_POLY = 0x8005
CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]  # x^8 + x^2 + x + 1
CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1]  # CRC-16-IBM


def _crc_remainder_poly(bits, generator):
    n = len(generator) - 1
    regs = [int(b) for b in bits] + [0] * n
    for i in range(len(bits)):
        if regs[i]:
            for j in range(len(generator)):
                regs[i + j] ^= generator[j]
    return regs[len(bits):len(bits) + n]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        gen = CRC8_GEN
    elif crc_length == 16:
        gen = CRC16_GEN
    else:
        raise ValueError("crc_length must be 8 or 16")

    crc_bits = np.array(_crc_remainder_poly(info_bits, gen), dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        gen = CRC8_GEN
    elif crc_length == 16:
        gen = CRC16_GEN
    else:
        raise ValueError("crc_length must be 8 or 16")

    n = crc_length
    regs = [int(b) for b in bits]
    for i in range(len(bits) - n):
        if regs[i]:
            for j in range(len(gen)):
                regs[i + j] ^= gen[j]
    return all(x == 0 for x in regs[-n:])


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.br = bit_reversal_permutation(N)

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _crc_pass(self, u_hat):
        if self.crc_length == 0:
            return True
        info_bits = u_hat[self.info_indices]
        return crc_check(info_bits, self.crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        llr_perm = llr_ch[self.br]

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_perm
        paths.append({"L": L0, "B": B0, "pm": 0.0})

        for l in [_bit_reversed(i, n) for i in range(N)]:
            new_paths = []
            for path in paths:
                L, B, pm = path["L"], path["B"], path["pm"]
                self._update_llrs(L, B, l)
                llr_val = L[l, n]

                if l in self.frozen_set:
                    new_pm = pm + (abs(llr_val) if llr_val < 0 else 0.0)
                    B[l, n] = 0
                    self._update_bits(B, l)
                    new_paths.append({"L": L, "B": B, "pm": new_pm})
                else:
                    for bit in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        new_pm = pm
                        if bit != (0 if llr_val >= 0 else 1):
                            new_pm += abs(llr_val)
                        Bc[l, n] = bit
                        self._update_bits(Bc, l)
                        new_paths.append({"L": Lc, "B": Bc, "pm": new_pm})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:self.list_size]

        candidates = []
        for path in paths:
            u_hat = path["B"][:, n].astype(int)
            candidates.append((path["pm"], u_hat))

        crc_ok = [(pm, u) for pm, u in candidates if self._crc_pass(u)]
        if crc_ok:
            pm, u_hat = min(crc_ok, key=lambda x: x[0])
        else:
            pm, u_hat = min(candidates, key=lambda x: x[0])

        return u_hat, pm


def verify_scl_equals_sc(N=64, K=32):
    """L=1 时 SCL 应等价于 SC"""
    from construction import ga_construction
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        from encoder import polar_encode
        from channel import bpsk_modulate, compute_llr
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
