"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特末尾。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) & mask) ^ poly
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    """单条 SCL 路径（Lazy Copy：分裂时浅拷贝数组引用，更新时复制受影响的切片）。"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)
        self.pm = 0.0

    def clone(self):
        p = _Path.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 风格路径复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        fb = np.asarray(frozen_bits)
        if fb.dtype == bool:
            self.frozen_bits = fb
        else:
            self.frozen_bits = fb.astype(int).astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if list_size == 1 and crc_length == 0:
            self._use_sc = True
        else:
            self._use_sc = False

    def _update_llrs(self, path, l_nat):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l_nat, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l_nat, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l_nat):
        n, N = self.n, self.N
        if l_nat < N // 2:
            return
        for s in range(n, n - _active_bit_level(l_nat, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l_nat, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        """与 LLR 不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self._use_sc:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l_nat = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l_nat)
                llr_bit = path.L[l_nat, self.n]

                if self.frozen_bits[l_nat]:
                    new_p = path.clone()
                    new_p.pm += self._pm_penalty(llr_bit, 0)
                    new_p.u_hat[l_nat] = 0
                    new_p.B[l_nat, self.n] = 0
                    self._update_bits(new_p, l_nat)
                    candidates.append(new_p)
                else:
                    for bit in (0, 1):
                        new_p = path.clone()
                        new_p.pm += self._pm_penalty(llr_bit, bit)
                        new_p.u_hat[l_nat] = bit
                        new_p.B[l_nat, self.n] = bit
                        self._update_bits(new_p, l_nat)
                        candidates.append(new_p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            passed = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            if passed:
                best = min(passed, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
