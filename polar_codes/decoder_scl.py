"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _frozen_set_from_mask,
)


# ==================== CRC 工具 ====================

def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0 or len(bits) < crc_length:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


def _update_llrs_path(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top = L[j - branch_size, s]
                bot = L[j, s]
                b = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(top, bot, b)


def _update_bits_path(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时深拷贝 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        N, n = self.N, self.n
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        L = np.zeros((N, n + 1), dtype=np.float64)
        B = np.zeros((N, n + 1), dtype=int)
        L[:, 0] = llr_ch
        paths = [(L, B, 0.0)]

        for i in range(N):
            l = bit_reversed(i, n)
            candidates = []
            for Lp, Bp, pm in paths:
                _update_llrs_path(Lp, Bp, l, n, N)
                llr = Lp[l, n]
                if l in self.frozen_set:
                    new_pm = pm + self._path_penalty(llr, 0)
                    Bp[l, n] = 0
                    _update_bits_path(Bp, l, n, N)
                    candidates.append((Lp, Bp, new_pm))
                else:
                    for u in (0, 1):
                        Lc = Lp.copy()
                        Bc = Bp.copy()
                        new_pm = pm + self._path_penalty(llr, u)
                        Bc[l, n] = u
                        _update_bits_path(Bc, l, n, N)
                        candidates.append((Lc, Bc, new_pm))

            candidates.sort(key=lambda x: x[2])
            paths = candidates[:self.list_size]

        best = None
        if self.crc_length > 0:
            for Lp, Bp, pm in paths:
                u_hat = Bp[:, n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best is None or pm < best[2]:
                        best = (Lp, Bp, pm)
        if best is None:
            best = paths[0]

        u_hat = best[1][:, n].astype(int)
        return u_hat, best[2]
