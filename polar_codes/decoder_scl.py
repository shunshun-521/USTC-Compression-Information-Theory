"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level,
    _prepare_llr,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in np.asarray(info_bits, dtype=int):
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _clone_path(self, path):
        return {
            'pm': path['pm'],
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'u_hat': path['u_hat'].copy(),
        }

    def decode(self, llr_ch):
        likelihoods = _prepare_llr(llr_ch)
        N, n = self.N, self.n

        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = likelihoods
        paths = [{'pm': 0.0, 'L': L0, 'B': B0, 'u_hat': np.zeros(N, dtype=int)}]

        for l in [_bit_reversed(i, n) for i in range(N)]:
            new_paths = []
            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr = path['L'][l, n]

                if l in self.frozen_set:
                    child = self._clone_path(path)
                    child['pm'] += self._metric_penalty(llr, 0)
                    child['B'][l, n] = 0
                    child['u_hat'][l] = 0
                    self._update_bits(child['B'], l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child['pm'] += self._metric_penalty(llr, bit)
                        child['B'][l, n] = bit
                        child['u_hat'][l] = bit
                        self._update_bits(child['B'], l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']


def verify_scl_equals_sc(N=64, K=32, num_frames=20):
    """L=1 时 SCL 应与 SC 等价"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = 0.1
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True


if __name__ == "__main__":
    print("SCL L=1 vs SC:", "PASSED" if verify_scl_equals_sc() else "FAILED")
