"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
    pm_penalty,
    _frozen_mask,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005

    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class _SCLPath:
    """单条 SCL 路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = _frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        L, B = path.L, path.B
        n = self.n
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        B = path.B
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _extract_u_hat(self, path):
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            u_hat[i] = path.B[l, self.n]
        return u_hat

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [_SCLPath(N, n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                cur_llr = path.L[l, n]

                if self.frozen[i]:
                    path.pm += pm_penalty(cur_llr, 0)
                    path.B[l, n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_cand in (0, 1):
                        p_copy = _SCLPath(N, n)
                        p_copy.L = path.L.copy()
                        p_copy.B = path.B.copy()
                        p_copy.pm = path.pm + pm_penalty(cur_llr, u_cand)
                        p_copy.B[l, n] = u_cand
                        self._update_bits(p_copy, l)
                        new_paths.append(p_copy)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = float("inf")
        best_u = None

        for path in paths:
            u_hat = self._extract_u_hat(path)
            if self.crc_length > 0:
                info_idx = np.where(~self.frozen)[0]
                payload = u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    if path.pm < best_pm:
                        best_pm = path.pm
                        best_u = u_hat
                        best_crc = True
            else:
                if path.pm < best_pm:
                    best_pm = path.pm
                    best_u = u_hat

        if best_u is None:
            best_u = self._extract_u_hat(paths[0])
            best_pm = paths[0].pm

        return best_u, best_pm
