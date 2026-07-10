"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _f_boxplus,
    _lower_llr,
    _to_frozen_set,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int64,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC 约束。"""
    bits = np.asarray(bits, dtype=np.int64)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0

class _PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1))


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _to_frozen_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            sorted(set(range(N)) - self.frozen_set), dtype=np.int64
        )

    def _path_metric_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _advance_path_to_bit(self, path, bit_index):
        """将单条路径推进到 bit_index 并返回该比特 LLR。"""
        n = self.n
        N = self.N
        L = path.L
        B = path.B
        l = _bit_reversed_index(bit_index, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

        return L[l, n]

    def _propagate_bit(self, path, bit_index, u_bit):
        n = self.n
        N = self.N
        B = path.B
        l = _bit_reversed_index(bit_index, n)
        B[l, n] = u_bit

        if l < N / 2:
            return

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        rev = bit_reversal_permutation(N)
        llr_perm = llr_ch[rev]

        paths = [_PathState(N, n)]
        paths[0].L[:, 0] = llr_perm

        for bit_index in range(N):
            l = _bit_reversed_index(bit_index, n)
            candidates = []

            for path in paths:
                llr_val = self._advance_path_to_bit(path, bit_index)
                branches = [0] if l in self.frozen_set else [0, 1]

                for u_bit in branches:
                    new_path = _PathState(N, n)
                    new_path.pm = path.pm + self._path_metric_penalty(llr_val, u_bit)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    self._propagate_bit(new_path, bit_index, u_bit)
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        decoded = []
        for path in paths:
            u_hat = path.B[:, n].astype(np.int64)
            decoded.append((u_hat, path.pm))

        if self.crc_length > 0:
            valid = [
                (u, pm)
                for u, pm in decoded
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                best_u, best_pm = min(valid, key=lambda x: x[1])
            else:
                best_u, best_pm = min(decoded, key=lambda x: x[1])
        else:
            best_u, best_pm = min(decoded, key=lambda x: x[1])

        return best_u, best_pm


def verify_scl_equals_sc(N=64, K=32, num_frames=50, seed=1):
    """L=1 的 SCL 应与 SC 等价。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(4.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=np.int64)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("SCL(L=1) 与 SC 不一致")

    return True


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL 路径度量校验通过")
