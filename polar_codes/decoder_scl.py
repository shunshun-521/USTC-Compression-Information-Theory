"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_mod(bits, poly, crc_len):
    reg = 0
    mask = (1 << crc_len) - 1
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位（CRC-8: 0x07, CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY if crc_length == 16 else None
    if poly is None:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_mod(np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY if crc_length == 16 else None
    if poly is None:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_mod(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch.copy()
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    def _update_bits(self, path, l):
        B = path.B
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p2 = _Path(self.N, self.n, llr_ch)
                        p2.L = path.L.copy()
                        p2.B = path.B.copy()
                        p2.pm = path.pm + self._pm_penalty(llr, u)
                        p2.u_hat = path.u_hat.copy()
                        p2.B[l, self.n] = u
                        p2.u_hat[l] = u
                        self._update_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = paths
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            passed = []
            for p in candidates:
                payload = p.u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    passed.append(p)
            if passed:
                candidates = passed

        best = min(candidates, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
