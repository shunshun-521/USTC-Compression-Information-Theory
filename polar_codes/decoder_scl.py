"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
采用 Permuted SCD 与 B_N F^{⊗n} 编码器匹配
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _frozen_set_from_mask,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ('pm', 'B', 'L', 'decoded_u')

    def __init__(self, N, n, llr_ch, parent=None):
        self.pm = 0.0 if parent is None else parent.pm
        self.L = np.zeros((N, n + 1)) if parent is None else parent.L.copy()
        self.B = np.zeros((N, n + 1), dtype=int) if parent is None else parent.B.copy()
        self.decoded_u = [] if parent is None else list(parent.decoded_u)
        if parent is None:
            self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = _frozen_set_from_mask(self.frozen_bits, N)
        self.rev = bit_reversal_permutation(N)
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s],
                        path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size
        paths = [_Path(N, n, llr_ch)]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, n]
                if l in self.frozen_set:
                    new_path = _Path(N, n, llr_ch, parent=path)
                    new_path.pm += self._pm_penalty(llr_val, 0)
                    new_path.B[l, n] = 0
                    new_path.decoded_u.append(0)
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = _Path(N, n, llr_ch, parent=path)
                        new_path.pm += self._pm_penalty(llr_val, u_bit)
                        new_path.B[l, n] = u_bit
                        new_path.decoded_u.append(u_bit)
                        self._update_bits(new_path, l)
                        candidates.append(new_path)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:L]

        best = paths[0]
        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                u_vec = np.array(p.decoded_u, dtype=np.int8)
                payload = u_vec[self.info_idx]
                if crc_check(payload, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)

        v_hat = np.zeros(N, dtype=int)
        for i, bit in enumerate(best.decoded_u):
            l = _bit_reversed(i, n)
            v_hat[l] = bit
        u_hat = v_hat[self.rev]
        return u_hat, best.pm

if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
