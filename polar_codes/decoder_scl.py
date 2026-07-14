"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class _PathState:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _branch_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = [_PathState(N, n) for _ in range(L_size)]
        paths[0].L[:, 0] = llr_ch
        active = [0]

        for phi_nat in range(N):
            l = _bit_reversed(phi_nat, n)
            candidates = []

            for pidx in active:
                path = paths[pidx]
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if self.frozen_bits[phi_nat]:
                    pm = path.pm + self._branch_penalty(llr, 0)
                    candidates.append((pm, pidx, 0))
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._branch_penalty(llr, bit)
                        candidates.append((pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            new_active = []

            for rank, (pm, pidx, bit) in enumerate(candidates[:L_size]):
                if rank >= len(active):
                    paths[rank] = _PathState(N, n)
                    paths[rank].L[:, 0] = llr_ch

                if rank < len(active) and active[rank] == pidx:
                    path = paths[rank]
                else:
                    path = paths[rank]
                    path.L = paths[pidx].L.copy()
                    path.B = paths[pidx].B.copy()
                    path.u_hat = paths[pidx].u_hat.copy()

                path.pm = pm
                path.u_hat[phi_nat] = bit
                path.B[l, n] = bit
                self._update_bits(path, l)
                new_active.append(rank)

            active = new_active

        best_idx = active[0]
        best_pm = paths[best_idx].pm
        crc_candidates = []

        for pidx in active:
            pm = paths[pidx].pm
            u_hat = paths[pidx].u_hat
            if self.crc_length > 0:
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_candidates.append((pm, pidx))
            if pm < best_pm:
                best_pm = pm
                best_idx = pidx

        if crc_candidates:
            crc_candidates.sort(key=lambda x: x[0])
            best_idx = crc_candidates[0][1]
            best_pm = crc_candidates[0][0]

        return paths[best_idx].u_hat.copy(), best_pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, 0.5)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
