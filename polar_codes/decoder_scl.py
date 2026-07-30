"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import (
    _bit_rev,
    _active_llr_level,
    _active_bit_level,
    _hard_decision,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_remainder(bits, crc_length):
    """GF(2) 多项式长除法求 CRC 余数。"""
    bits = list(np.asarray(bits, dtype=int))
    poly = _crc_poly(crc_length)
    poly_bits = []
    p = poly | (1 << crc_length)
    while p:
        poly_bits.append(p & 1)
        p >>= 1
    poly_bits = poly_bits[::-1]

    msg = bits + [0] * crc_length
    for i in range(len(bits)):
        if msg[i] == 1:
            for j in range(len(poly_bits)):
                msg[i + j] ^= poly_bits[j]
    return int("".join(map(str, msg[-crc_length:])), 2)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    return _crc_remainder(bits, crc_length) == 0


class _Path:
    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.pm = 0.0

    def clone(self):
        p = _Path(self.N, self.n)
        p.L[:] = self.L
        p.B[:] = self.B
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.brev = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_add(self, pm, llr, bit):
        expected = _hard_decision(llr)
        if bit != expected:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_mapped = llr_ch[self.brev]

        root = _Path(self.N, self.n)
        root.L[:, 0] = llr_mapped
        paths = [root]

        for i in range(self.N):
            l = _bit_rev(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    child = path.clone()
                    child.pm = self._pm_add(path.pm, llr, 0)
                    child.B[l, self.n] = 0
                    self._update_bits(child, l)
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = path.clone()
                        child.pm = self._pm_add(path.pm, llr, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        best_pm = float("inf")
        best_u = None
        crc_pass_pm = float("inf")
        crc_pass_u = None

        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            pm = path.pm
            if pm < best_pm:
                best_pm = pm
                best_u = u_hat.copy()

            if self.crc_length > 0:
                info_idx = np.where(self.frozen_bits == 0)[0]
                payload = u_hat[info_idx]
                if crc_check(payload, self.crc_length) and pm < crc_pass_pm:
                    crc_pass_pm = pm
                    crc_pass_u = u_hat.copy()

        if crc_pass_u is not None:
            return crc_pass_u, crc_pass_pm
        return best_u, best_pm
