"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加（MSB 先传输）。"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=np.int8):
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "parent_id")

    def __init__(self, N, n, llr_ch, path_id):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.parent_id = path_id


class SCLDecoder:
    """SCL 译码器（permuted SC 顺序，路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits(self, path, l):
        B = path.B
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u_bit):
        """与 LLR 不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch, 0)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for p_idx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr_leaf = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_leaf, 0)
                    path.pm += pen
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append((path.pm, p_idx, path, None))
                else:
                    for u_bit in (0, 1):
                        new_path = _Path(self.N, self.n, llr_ch, p_idx)
                        new_path.L[:] = path.L
                        new_path.B[:] = path.B
                        new_path.u_hat[:] = path.u_hat
                        new_path.pm = path.pm + self._pm_penalty(llr_leaf, u_bit)
                        new_path.u_hat[l] = u_bit
                        new_path.B[l, self.n] = u_bit
                        self._update_bits(new_path, l)
                        candidates.append((new_path.pm, p_idx, new_path, u_bit))

            candidates.sort(key=lambda x: x[0])
            paths = [c[2] for c in candidates[: self.list_size]]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
