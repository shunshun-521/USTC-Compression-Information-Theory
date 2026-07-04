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
    f_boxplus,
    g_operation,
    precompute_sc_indices,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _pm_update(self, pm, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        if u_val != hard:
            pm += abs(llr_val)
        return pm

    def _update_llrs(self, llr_mem, bit_mem, phi):
        l = _bit_reversed(phi, self.n)
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    llr_mem[j, s + 1] = f_boxplus(llr_mem[j, s], llr_mem[j + branch_size, s])
                else:
                    llr_mem[j, s + 1] = g_operation(
                        llr_mem[j - branch_size, s],
                        llr_mem[j, s],
                        bit_mem[j - branch_size, s + 1],
                    )
        return llr_mem[l, self.n]

    def _update_bits(self, bit_mem, phi, u_val):
        l = _bit_reversed(phi, self.n)
        bit_mem[l, self.n] = u_val
        if l >= self.N / 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        bit_mem[j - branch_size, s - 1] = (
                            bit_mem[j, s] ^ bit_mem[j - branch_size, s]
                        )
                        bit_mem[j, s - 1] = bit_mem[j, s]

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        N = self.N
        n = self.n

        paths = [{
            "pm": 0.0,
            "llr": np.full((N, n + 1), np.nan, dtype=np.float64),
            "bit": np.zeros((N, n + 1), dtype=np.int8),
            "u": np.zeros(N, dtype=int),
        }]
        paths[0]["llr"][:, 0] = llr

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for pidx, path in enumerate(paths):
                cur_llr = self._update_llrs(path["llr"], path["bit"], phi)
                if l in self.frozen_set:
                    candidates.append((self._pm_update(path["pm"], cur_llr, 0), pidx, 0))
                else:
                    for u_val in (0, 1):
                        pm = self._pm_update(path["pm"], cur_llr, u_val)
                        candidates.append((pm, pidx, u_val))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, u_val in candidates:
                parent = paths[pidx]
                llr_copy = parent["llr"].copy()
                bit_copy = parent["bit"].copy()
                u_copy = parent["u"].copy()
                u_copy[l] = u_val
                self._update_bits(bit_copy, phi, u_val)
                new_paths.append({
                    "pm": pm,
                    "llr": llr_copy,
                    "bit": bit_copy,
                    "u": u_copy,
                })
            paths = new_paths

        best_u = None
        best_pm = float("inf")

        if self.crc_length > 0:
            crc_pass = []
            for path in paths:
                info_bits = path["u"][~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            pool = crc_pass if crc_pass else paths
        else:
            pool = paths

        for path in pool:
            if path["pm"] < best_pm:
                best_pm = path["pm"]
                best_u = path["u"]

        return best_u, best_pm
