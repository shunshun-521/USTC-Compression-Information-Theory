"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversed
from decoder_sc import (
  _active_llr_level, _active_bit_level, _upper_llr, _lower_llr, _hard_decision,
)


# ==================== CRC 工具 ====================

_POLY8 = [1, 0, 0, 0, 0, 0, 1, 1, 1]          # x^8 + x^2 + x + 1
_POLY16 = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # CRC-16-IBM


def _gf2_remainder(msg, poly):
    """GF(2) 多项式长除余数，MSB 在前"""
    msg = list(map(int, msg))
    poly = list(map(int, poly))
    n = len(poly) - 1
    for i in range(len(msg) - n):
        if msg[i] == 1:
            for j in range(len(poly)):
                msg[i + j] ^= poly[j]
    return np.array(msg[-n:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _POLY8 if crc_length == 8 else _POLY16
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _gf2_remainder(padded, poly)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _POLY8 if crc_length == 8 else _POLY16
    rem = _gf2_remainder(bits, poly)
    return np.all(rem == 0)


def _pm_update(pm, llr, u):
    """路径度量更新：与 LLR 符号不一致时加 |LLR|"""
    expected = 0 if llr >= 0 else 1
    if u != expected:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_paths(self, llr_ch):
        paths = [{
            'L': np.zeros((self.N, self.n + 1), dtype=np.float64),
            'B': np.zeros((self.N, self.n + 1), dtype=np.int32),
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch
        return paths

    def _update_llrs_path(self, path, l):
        L = path['L']
        B = path['B']
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits_path(self, path, l):
        B = path['B']
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs_path(path, l)
                llr_bit = path['L'][l, self.n]

                if l in self.frozen_set:
                    pm = _pm_update(path['pm'], llr_bit, 0)
                    child = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': pm,
                        'u_hat': path['u_hat'].copy(),
                    }
                    child['B'][l, self.n] = 0
                    child['u_hat'][l] = 0
                    self._update_bits_path(child, l)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = _pm_update(path['pm'], llr_bit, u)
                        child = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': pm,
                            'u_hat': path['u_hat'].copy(),
                        }
                        child['B'][l, self.n] = u
                        child['u_hat'][l] = u
                        self._update_bits_path(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            info_idx = [i for i in range(self.N) if i not in self.frozen_set]
            valid = []
            for p in paths:
                info_bits = p['u_hat'][info_idx]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'], best['pm']


def verify_scl_equals_sc(N=64, K=32, num_frames=20):
    """L=1 时 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(3.0, K / N)
    rng = np.random.default_rng(1)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print(f"SCL(L=1) 验证通过: N={N}, K={K}")
