"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    _prepare_channel_llr,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.uint8)
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

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits.astype(int), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat", "active")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int_)
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组引用，更新时按需复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            [i for i in range(N) if i not in self.frozen_set], dtype=int
        )

    def _path_llr(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], path.B[j - half, s + 1]
                    )

    def _path_update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 2 ** s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.B[j - half, s - 1] = path.B[j, s] ^ path.B[j - half, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr0 = _prepare_channel_llr(llr_ch, self.N)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr0

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._path_llr(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    bit = 0
                    new_pm = path.pm + self._pm_penalty(llr, bit)
                    path.pm = new_pm
                    path.B[l, self.n] = bit
                    path.u_hat[l] = bit
                    self._path_update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        p = _Path(self.N, self.n)
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.u_hat = path.u_hat.copy()
                        p.pm = path.pm + self._pm_penalty(llr, bit)
                        p.B[l, self.n] = bit
                        p.u_hat[l] = bit
                        self._path_update_bits(p, l)
                        candidates.append(p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        valid = []
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)

        best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
