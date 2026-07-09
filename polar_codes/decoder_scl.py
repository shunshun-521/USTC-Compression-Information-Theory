"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class PathState:
    __slots__ = ("pm", "L", "C", "u_hat", "active")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.C = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = False


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/C）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, path, phi):
        n, N = self.n, self.N
        l = bit_reversed(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], path.C[j - half, s + 1]
                    )

    def _update_bits(self, path, phi):
        n, N = self.n, self.N
        l = bit_reversed(phi, n)
        if l < N // 2:
            return
        end = n - _active_bit_level(l, n)
        for s in range(n, end, -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.C[j - half, s - 1] = path.C[j, s] ^ path.C[j - half, s]
                    path.C[j, s - 1] = path.C[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N, L = self.n, self.N, self.list_size

        paths = [PathState(N, n) for _ in range(L)]
        paths[0].active = True
        paths[0].L[:, 0] = llr_ch

        for phi in range(N):
            l = bit_reversed(phi, n)
            candidates = []

            for p in paths:
                if not p.active:
                    continue
                self._update_llrs(p, phi)
                llr = p.L[l, n]
                if self.frozen_bits[l]:
                    candidates.append((p.pm + self._metric_penalty(llr, 0), p, 0))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (p.pm + self._metric_penalty(llr, bit), p, bit)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L]

            new_paths = [PathState(N, n) for _ in range(L)]
            for i, (pm, old, bit) in enumerate(candidates):
                new_paths[i].pm = pm
                new_paths[i].L = old.L.copy()
                new_paths[i].C = old.C.copy()
                new_paths[i].u_hat = old.u_hat.copy()
                new_paths[i].active = True
                new_paths[i].C[l, n] = bit
                new_paths[i].u_hat[l] = bit
                self._update_bits(new_paths[i], phi)

            paths = new_paths

        valid = [(p.pm, p.u_hat) for p in paths if p.active]

        if self.crc_length > 0:
            crc_ok = []
            for pm, u in valid:
                if crc_check(u[self.info_indices], self.crc_length):
                    crc_ok.append((pm, u))
            if crc_ok:
                pm, best = min(crc_ok, key=lambda x: x[0])
                return best.copy(), pm

        pm, best = min(valid, key=lambda x: x[0])
        return best.copy(), pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(10.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
