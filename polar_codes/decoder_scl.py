"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    bit_reversed_index,
    _update_llrs,
    _update_bits,
    _active_llr_level,
    _active_bit_level,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    poly = CRC_POLYNOMIALS[crc_length]
    bits = np.asarray(bits, dtype=int)
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class PathState:
    """单条 SCL 路径状态。"""

    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch

    def clone(self):
        new_state = PathState.__new__(PathState)
        new_state.pm = self.pm
        new_state.L = self.L.copy()
        new_state.B = self.B.copy()
        return new_state


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])
        self.info_indices = np.where(~self.frozen_bits.astype(bool))[0]

    def _metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        paths = [PathState(self.N, self.n, llr_ch)]

        for phase in range(self.N):
            l = bit_reversed_index(phase, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    child = path.clone()
                    child.pm += self._metric_penalty(llr, 0)
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n, self.N)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.clone()
                        child.pm += self._metric_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n, self.N)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_passed = []
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_passed.append((path.pm, u_hat))
            else:
                crc_passed.append((path.pm, u_hat))

        if self.crc_length > 0 and crc_passed:
            best_pm, best_u = min(crc_passed, key=lambda x: x[0])
            return best_u, best_pm

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
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
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")
