"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _bit_reversed, _update_llrs, update_bits_for_sc

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.pm = 0.0

    def copy(self):
        p = _Path(self.L.shape[0], int(math.log2(self.L.shape[0])))
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（路径复制实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_add(self, llr, u):
        u_hat = 0 if llr >= 0 else 1
        return 0.0 if u == u_hat else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    p = path.copy()
                    p.pm += self._pm_add(llr, 0)
                    p.B[l, self.n] = 0
                    update_bits_for_sc(p.B, l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm += self._pm_add(llr, u)
                        p.B[l, self.n] = u
                        update_bits_for_sc(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info = p.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm


def verify_scl_equals_sc(N=64, frozen_bits=None, num_frames=20):
    """L=1 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma, map_llr_to_decoder
    from decoder_sc import sc_decode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    if frozen_bits is None:
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(4.0, 0.5)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = map_llr_to_decoder(
            compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
