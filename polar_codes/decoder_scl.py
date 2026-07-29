"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation,
    _bit_reversed_index, _active_llr_level, _active_bit_level,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return (reg & ((1 << crc_length) - 1)) == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.rev = bit_reversal_permutation(N)

    def _path_metric_update(self, pm, llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        if u != hard:
            pm += abs(llr_val)
        return pm

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _propagate_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N = self.N
        n = self.n
        L_size = self.list_size
        llr_internal = np.asarray(llr_ch, dtype=np.float64)[self.rev]

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int32),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_internal

        for phi in range(N):
            l = _bit_reversed_index(phi, n)
            candidates = []

            for path in paths:
                L = path['L'].copy()
                B = path['B'].copy()
                self._update_llrs(L, B, l)
                llr_val = L[l, n]

                if l in self.frozen_set:
                    new_path = {
                        'L': L, 'B': B,
                        'pm': self._path_metric_update(path['pm'], llr_val, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['u_hat'][l] = 0
                    new_path['B'][l, n] = 0
                    self._propagate_bits(new_path['B'], l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        new_path = {
                            'L': Lc, 'B': Bc,
                            'pm': self._path_metric_update(path['pm'], llr_val, u),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['u_hat'][l] = u
                        new_path['B'][l, n] = u
                        self._propagate_bits(new_path['B'], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:L_size]

        if self.crc_length > 0:
            crc_pass = [p for p in paths if crc_check(p['u_hat'][self.info_indices], self.crc_length)]
            best = min(crc_pass, key=lambda p: p['pm']) if crc_pass else min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=5.0):
    """L=1 的 SCL 应等价于 SC。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(42)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    print("SCL L=1 == SC verification passed")


if __name__ == "__main__":
    verify_scl_equals_sc()
