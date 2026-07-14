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


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    reg = 0
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & (1 << (crc_length - 1)):
                    reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
                else:
                    reg = (reg << 1) & ((1 << crc_length) - 1)
    if crc_length == 8:
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时共享 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _llr_at(self, path, phi):
        l = _bit_reversed(phi, self.n)
        _update_llrs(path.L, path.B, l, self.n, self.N)
        return path.L[l, self.n], l

    def _propagate_bit(self, path, l, bit):
        path.B[l, self.n] = bit
        _update_bits(path.B, l, self.n, self.N)

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            candidates = []
            for pidx, path in enumerate(paths):
                llr, l = self._llr_at(path, phi)
                if self.frozen_bits[phi]:
                    pm_new = path.pm + self._pm_penalty(llr, 0)
                    candidates.append((pm_new, pidx, 0, llr, l))
                else:
                    for u in (0, 1):
                        pm_new = path.pm + self._pm_penalty(llr, u)
                        candidates.append((pm_new, pidx, u, llr, l))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm_new, pidx, u, llr, l in candidates[: self.list_size]:
                parent = paths[pidx]
                child = _Path(self.N, self.n, llr_ch)
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.pm = pm_new
                child.u_hat = parent.u_hat.copy()
                child.u_hat[phi] = u
                self._propagate_bit(child, l, u)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(8.0, K / N)
    mism = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mism += 1
    print(f"SCL L=1 vs SC mismatches: {mism}/50")
