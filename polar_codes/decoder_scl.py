"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_int,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (fb * poly)
    crc_val = reg
    crc_bits = np.array([(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ (fb * poly)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD 框架）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_indices = set(np.where(self.frozen_bits)[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _llr_root(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1])
        return L[l, self.n]

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.br]

        paths = [
            {
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_br

        for i in range(self.N):
            l = _bit_reversed_int(i, self.n)
            cand = []
            for p_idx, path in enumerate(paths):
                L = path["L"]
                B = path["B"]
                llr = self._llr_root(L, B, l)
                if l in self.frozen_indices:
                    pen = 0.0 if llr >= 0 else abs(llr)
                    cand.append((path["pm"] + pen, p_idx, 0, llr))
                else:
                    for u_bit in (0, 1):
                        hard = 0 if llr >= 0 else 1
                        pen = 0.0 if u_bit == hard else abs(llr)
                        cand.append((path["pm"] + pen, p_idx, u_bit, llr))

            cand.sort(key=lambda x: x[0])
            cand = cand[: self.list_size]

            new_paths = []
            for pm, parent_idx, u_bit, _ in cand:
                parent = paths[parent_idx]
                child = {
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "pm": pm,
                }
                child["B"][l, self.n] = u_bit
                self._update_bits(child["B"], l)
                new_paths.append(child)
            paths = new_paths

        best = None
        crc_candidates = []
        for path in paths:
            u_hat = path["B"][:, self.n].astype(int)
            if best is None or path["pm"] < best[0]:
                best = (path["pm"], u_hat)
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_candidates.append((path["pm"], u_hat))

        if self.crc_length > 0 and crc_candidates:
            crc_candidates.sort(key=lambda x: x[0])
            return crc_candidates[0][1], crc_candidates[0][0]
        return best[1], best[0]
