"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    reorder_llr_for_decoder,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    if crc_length == 8:
        poly = 0x07
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = 0x8005
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


def _pm_update(pm, llr, u_bit):
    """路径度量：与 LLR 符号不一致时加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u")

    def __init__(self, N, n, llr):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制实现，列表较小时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _advance_paths(self, paths, l):
        """对当前比特索引 l 扩展所有路径"""
        new_paths = []
        for path in paths:
            self._update_llrs(path, l)
            llr = path.L[l, self.n]
            if l in self.frozen_set:
                path.u[l] = 0
                path.B[l, self.n] = 0
                path.pm = _pm_update(path.pm, llr, 0)
                self._update_bits(path, l)
                new_paths.append(path)
            else:
                for u_bit in (0, 1):
                    cp = _Path(self.N, self.n, path.L[:, 0].copy())
                    cp.L = path.L.copy()
                    cp.B = path.B.copy()
                    cp.pm = path.pm
                    cp.u = path.u.copy()
                    cp.u[l] = u_bit
                    cp.B[l, self.n] = u_bit
                    cp.pm = _pm_update(cp.pm, llr, u_bit)
                    self._update_bits(cp, l)
                    new_paths.append(cp)
        new_paths.sort(key=lambda p: p.pm)
        return new_paths[: self.list_size]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        int(path.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr = reorder_llr_for_decoder(llr_ch)
        paths = [_Path(self.N, self.n, llr)]
        order = [_bit_reversed(i, self.n) for i in range(self.N)]
        for l in order:
            paths = self._advance_paths(paths, l)

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_pass(p)]
            candidates = valid if valid else paths
        else:
            candidates = paths
        best = min(candidates, key=lambda p: p.pm)
        return best.u.copy(), best.pm

    def _crc_pass(self, path):
        info_idx = np.where(~self.frozen_bits)[0]
        payload = path.u[info_idx]
        if len(payload) < self.crc_length:
            return False
        return crc_check(payload, self.crc_length)
