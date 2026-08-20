"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _prepare_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError('crc_length must be 8 or 16')


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if crc_length == 8:
                msb = (reg >> 7) & 1
                reg = ((reg << 1) & 0xFF) ^ (poly if msb else 0)
            else:
                msb = (reg >> 15) & 1
                reg = ((reg << 1) & 0xFFFF) ^ (poly if msb else 0)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(crc_length):
            msb = (reg >> (crc_length - 1)) & 1
            reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (poly if msb else 0)
    return reg == 0


def _llr_to_bit(llr):
    return 0 if llr >= 0 else 1


def _path_metric_update(pm, llr, u):
    v = _llr_to_bit(llr)
    if u == v:
        return pm
    return pm + abs(llr)


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径对象仅在分裂时复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen = set(np.where(frozen_bits)[0])
        self.info_set = sorted(set(range(N)) - self.frozen)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = 0 if np.isnan(path.B[j - branch_size, s + 1]) else int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(path.L[j - branch_size, s], path.L[j, s], top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr, _ = _prepare_llr(llr_ch)

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                cur_llr = path.L[l, self.n]

                if l in self.frozen:
                    new_path = _Path(self.N, self.n)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.u_hat = path.u_hat.copy()
                    new_path.pm = _path_metric_update(path.pm, cur_llr, 0)
                    new_path.u_hat[l] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = _Path(self.N, self.n)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = _path_metric_update(path.pm, cur_llr, u)
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[l] = u
                        new_path.B[l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_set]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=10.0, seed=0):
    """L=1 的 SCL 应与 SC 等价。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.zeros(N, dtype=int)
    frozen_bits[np.setdiff1d(np.arange(N), info_idx)] = 1

    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(seed)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_scl, u_sc), 'L=1 SCL 与 SC 不一致'
    return True
