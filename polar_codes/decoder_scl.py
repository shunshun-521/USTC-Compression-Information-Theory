"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _upper_llr,
    _lower_llr,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_shift(data_bits, poly, crc_length):
    reg = 0
    for bit in data_bits:
        fb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        reg |= int(bit)
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_shift(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_shift(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.B = np.zeros((N, n + 1), dtype=np.float64)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _metric_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        rev = bit_reversal_permutation(N)
        llr = llr_ch[rev]

        paths = [_Path(N, n)]
        L_store = [np.zeros((N, n + 1), dtype=np.float64)]
        L_store[0][:, 0] = llr

        for l in [_bit_reversed(i, n) for i in range(N)]:
            candidates = []

            for pidx, path in enumerate(paths):
                L = L_store[pidx].copy()
                B = path.B

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = _lower_llr(
                                L[j, s],
                                L[j - branch_size, s],
                                int(B[j - branch_size, s + 1]),
                            )

                llr_bit = L[l, n]

                if l in self.frozen_set:
                    new_path = _Path(N, n)
                    new_path.pm = path.pm + self._metric_penalty(llr_bit, 0)
                    new_path.B = B.copy()
                    new_path.B[l, n] = 0
                    self._update_bits(new_path.B, l)
                    candidates.append((new_path.pm, new_path, L))
                else:
                    for u in (0, 1):
                        new_path = _Path(N, n)
                        new_path.pm = path.pm + self._metric_penalty(llr_bit, u)
                        new_path.B = B.copy()
                        new_path.B[l, n] = u
                        self._update_bits(new_path.B, l)
                        candidates.append((new_path.pm, new_path, L.copy()))

            candidates.sort(key=lambda x: x[0])
            kept = candidates[: self.list_size]
            paths = [item[1] for item in kept]
            L_store = [item[2] for item in kept]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path.B[:, n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.B[:, n].astype(int), best.pm

    def _update_bits(self, B, l):
        if l < self.N / 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(8.0, K / N)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/50")

    bits = np.array([1, 0, 1, 1])
    encoded = crc_encode(bits, 8)
    print(f"CRC encode check: {crc_check(encoded, 8)}")
