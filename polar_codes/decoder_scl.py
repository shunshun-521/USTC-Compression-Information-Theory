"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed, bit_reversal_permutation
from decoder_sc import (
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
    f_operation,
    g_operation,
)


# CRC-8 (0x07), CRC-16 (0x8005)
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for bit in info_bits:
            reg ^= (bit << 7)
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for bit in info_bits:
            reg ^= bit << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    """SCL 单条路径（Lazy Copy：共享 LLR/比特数组引用）"""
    __slots__ = ("pm", "L", "B", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_set = set(np.where(~self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, u):
        """路径度量更新"""
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def _update_llrs_path(self, path, l):
        L, B = path.L, path.B
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = int(2 ** (s + 1))
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], top_bit
                    )

    def _update_bits_path(self, path, l):
        B = path.B
        n = self.n
        N = self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = int(2**s)
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _clone_path(self, path):
        """浅拷贝路径（Lazy Copy）"""
        new = Path(self.N, self.n, path.L[:, 0].copy())
        new.pm = path.pm
        new.L = path.L.copy()
        new.B = path.B.copy()
        new.u_hat = path.u_hat.copy()
        return new

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = [Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs_path(path, l)
                cur_llr = path.L[l, self.n]
                if np.isnan(cur_llr):
                    cur_llr = path.L[l, self.n - 1] if self.n > 0 else 0.0

                if l in self.frozen_set:
                    pm = self._pm_update(path.pm, cur_llr, 0)
                    p = self._clone_path(path)
                    p.pm = pm
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    self._update_bits_path(p, l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = self._clone_path(path)
                        p.pm = self._pm_update(path.pm, cur_llr, u)
                        p.u_hat[l] = u
                        p.B[l, self.n] = u
                        self._update_bits_path(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L_size]

        # 选择最优路径
        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = sorted(self.info_set)
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_idx], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm


def verify_scl_equals_sc(N=64, K=32):
    """L=1 时 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(8, 0.5)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("SCL L=1 与 SC 不一致")
    return True
