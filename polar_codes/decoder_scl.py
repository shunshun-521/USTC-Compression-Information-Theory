"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level, _bit_reversed


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    poly = CRC_POLYNOMIALS[crc_length]
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class Path:
    """单条 SCL 译码路径（Lazy Copy）。"""

    __slots__ = ("L", "B", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        brev = bit_reversal_permutation(N)
        self.L[:, 0] = llr_ch[brev]
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

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
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l, bit):
        path.B[l, self.n] = bit
        path.u_hat[l] = bit
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

    def _path_metric_penalty(self, llr, bit):
        """与 SC 一致的路径度量惩罚。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            for path in paths:
                self._update_llrs(path, l)

            new_paths = []
            if self.frozen_bits[l]:
                for path in paths:
                    llr = path.L[l, self.n]
                    path.pm += self._path_metric_penalty(llr, 0)
                    self._update_bits(path, l, 0)
                    new_paths.append(path)
            else:
                for path in paths:
                    llr = path.L[l, self.n]
                    for bit in (0, 1):
                        child = type(path).__new__(type(path))
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = path.pm + self._path_metric_penalty(llr, bit)
                        child.active = True
                        self._update_bits(child, l, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p.u_hat)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _crc_valid(self, u_hat):
        """检查信息位中的 CRC 是否有效。"""
        info_positions = np.where(~self.frozen_bits)[0]
        if len(info_positions) < self.crc_length:
            return False
        payload = u_hat[info_positions]
        return crc_check(payload, self.crc_length)
