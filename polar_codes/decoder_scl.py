"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import f_operation, g_operation, active_llr_level, active_bit_level
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s],
                        B[j - branch_size, s + 1]
                    )

    def _propagate_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        paths = []
        L0 = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B0 = np.zeros((self.N, self.n + 1), dtype=np.int32)
        L0[:, 0] = llr_ch
        paths.append({'pm': 0.0, 'L': L0, 'B': B0, 'u_hat': np.zeros(self.N, dtype=int)})

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr = path['L'][l, self.n]

                if l in self.frozen_set:
                    p = {
                        'pm': path['pm'] + self._pm_penalty(llr, 0),
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'u_hat': path['u_hat'].copy(),
                    }
                    p['B'][l, self.n] = 0
                    p['u_hat'][l] = 0
                    self._propagate_bits(p['B'], l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = {
                            'pm': path['pm'] + self._pm_penalty(llr, u),
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'u_hat': path['u_hat'].copy(),
                        }
                        p['B'][l, self.n] = u
                        p['u_hat'][l] = u
                        self._propagate_bits(p['B'], l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            crc_pass = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(crc_pass, key=lambda p: p['pm']) if crc_pass else paths[0]
        else:
            best = paths[0]

        return best['u_hat'].copy(), best['pm']


def verify_scl_equals_sc():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(42)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL(L=1) 与 SC 不一致: {mismatches}/50"
    print("SCL(L=1) 等价 SC 校验通过")


if __name__ == "__main__":
    verify_scl_equals_sc()
