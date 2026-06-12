"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


_CRC_POLY = {
    8: [8, 2, 1, 0],
    16: [16, 15, 2, 0],
}


def _crc_poly_bits(crc_length):
    loc = _CRC_POLY[crc_length]
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def _crc_division(info_bits, crc_length):
    """GF(2) 多项式长除，返回余数校验位。"""
    p = _crc_poly_bits(crc_length)
    work = list(map(int, info_bits)) + [0] * crc_length
    times = len(info_bits)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[j + i] ^= p[j]
    return work[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    check = _crc_division(info_bits, crc_length)
    return np.concatenate([info_bits, np.array(check, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info_len = len(bits) - crc_length
    expected = crc_encode(bits[:info_len], crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _pm_add(self, pm, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        if u_bit != hard:
            pm += abs(llr_val)
        return pm

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        N, n = self.N, self.n
        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(l, path.L, path.B, n, N)
                llr_val = path.L[l, n]

                if self.frozen_bits[l]:
                    pm = self._pm_add(path.pm, llr_val, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u_bit in (0, 1):
                        pm = self._pm_add(path.pm, llr_val, u_bit)
                        candidates.append((pm, path, u_bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u_bit in candidates:
                child = _Path(N, n)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.u_hat[l] = u_bit
                child.B[l, n] = u_bit
                _update_bits(l, child.B, n, N)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        u_hat = best.u_hat.copy()
        u_hat[self.frozen_bits] = 0
        return u_hat, best.pm


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=12.0):
    """L=1 的 SCL 应与 SC 等价。"""
    from construction import ga_construction
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from encoder import polar_encode
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(1)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
