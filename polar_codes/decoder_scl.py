"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
)


def _crc_modulus(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加到信息比特后（MSB 先）"""
    poly = _crc_modulus(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8).tolist()
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([np.asarray(bits, dtype=np.int8), np.asarray(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    poly = _crc_modulus(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in np.asarray(bits, dtype=np.int8):
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制 + Permuted SCD 更新）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(frozen_bits.astype(bool))[0])
        self.info_indices = np.where(~frozen_bits.astype(bool))[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _crc_ok(self, u_hat):
        if self.crc_length <= 0:
            return True
        info_idx = self.info_indices
        payload = u_hat[info_idx]
        return crc_check(payload, self.crc_length)

    def decode(self, llr_ch):
        if self.list_size == 1:
            from decoder_sc import sc_decode

            frozen_bits = np.ones(self.N, dtype=int)
            frozen_bits[list(self.info_indices)] = 0
            u_hat = sc_decode(llr_ch, frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                is_frozen = l in self.frozen_set

                if is_frozen:
                    p = _Path(self.N, self.n)
                    p.L = path.L.copy()
                    p.B = path.B.copy()
                    p.pm = path.pm + self._pm_penalty(llr, 0)
                    p.u_hat = path.u_hat.copy()
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = _Path(self.N, self.n)
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.pm = path.pm + self._pm_penalty(llr, u)
                        p.u_hat = path.u_hat.copy()
                        p.B[l, self.n] = u
                        p.u_hat[l] = u
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_paths = [p for p in paths if self._crc_ok(p.u_hat)]
        best = min(crc_paths or paths, key=lambda p: p.pm)
        return best.u_hat.copy(), float(best.pm)
