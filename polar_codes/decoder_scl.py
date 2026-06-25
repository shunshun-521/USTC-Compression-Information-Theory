"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _prepare_channel_llr,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, channel_llr):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)
        self.L[:, 0] = channel_llr


class SCLDecoder:
    """
    SCL 译码器（路径分裂时 Lazy Copy L/B）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(self.frozen_bits == 0)[0]
        self.decode_order = [
            _bit_reversed_index(phi, self.n) for phi in range(self.N)
        ]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _update_llrs(self, l, path):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l, path):
        if l < self.N // 2:
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

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        channel_llr = _prepare_channel_llr(llr_ch)
        paths = [_Path(self.N, self.n, channel_llr)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(l, path)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._pm_penalty(llr_val, 0)
                    child = _Path(self.N, self.n, channel_llr)
                    child.pm = path.pm + penalty
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    self._update_bits(l, child)
                    candidates.append(child)
                else:
                    for u_bit in (0, 1):
                        penalty = self._pm_penalty(llr_val, u_bit)
                        child = _Path(self.N, self.n, channel_llr)
                        child.pm = path.pm + penalty
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = u_bit
                        child.B[l, self.n] = u_bit
                        self._update_bits(l, child)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat, path.pm

        best = paths[0]
        return best.u_hat, best.pm
