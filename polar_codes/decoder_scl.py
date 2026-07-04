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
    _prepare_llr,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_register(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_register(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_register(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "llr", "bits", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.llr = np.zeros((N, n + 1), dtype=np.float64)
        self.bits = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, leaf):
        for stage in range(self.n - _active_llr_level(leaf, self.n), self.n):
            block_size = 1 << (stage + 1)
            branch_size = block_size // 2
            for j in range(leaf, self.N, block_size):
                if j % block_size < branch_size:
                    path.llr[j, stage + 1] = f_operation(
                        path.llr[j, stage], path.llr[j + branch_size, stage]
                    )
                else:
                    path.llr[j, stage + 1] = g_operation(
                        path.llr[j, stage],
                        path.llr[j - branch_size, stage],
                        path.bits[j - branch_size, stage + 1],
                    )

    def _propagate_bits(self, path, leaf):
        if leaf < self.N // 2:
            return
        for stage in range(self.n, self.n - _active_bit_level(leaf, self.n), -1):
            block_size = 1 << stage
            branch_size = block_size // 2
            for j in range(leaf, -1, -block_size):
                if j % block_size >= branch_size:
                    path.bits[j - branch_size, stage - 1] = (
                        path.bits[j, stage] ^ path.bits[j - branch_size, stage]
                    )
                    path.bits[j, stage - 1] = path.bits[j, stage]

    @staticmethod
    def _pm_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n)]
        paths[0].llr[:, 0] = llr_ch

        for phi in range(self.N):
            leaf = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, leaf)
                cur_llr = path.llr[leaf, self.n]

                if leaf in self.frozen_set:
                    path.pm += self._pm_penalty(cur_llr, 0)
                    path.bits[leaf, self.n] = 0
                    path.u_hat[leaf] = 0
                    self._propagate_bits(path, leaf)
                    candidates.append(path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._pm_penalty(cur_llr, u_bit)
                        new_path.bits[leaf, self.n] = u_bit
                        new_path.u_hat[leaf] = u_bit
                        self._propagate_bits(new_path, leaf)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n)
        new_path.pm = path.pm
        new_path.llr = path.llr.copy()
        new_path.bits = path.bits.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path


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
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}"
    print("SCL decoder tests passed")
