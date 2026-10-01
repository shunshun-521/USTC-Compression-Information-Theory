"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
    __slots__ = ("pm", "u_hat")

    def __init__(self):
        self.pm = 0.0
        self.u_hat = None


class SCLDecoder:
    """SCL 译码器（基于因子图 LLR，路径分裂时复制 u_hat）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.large = 1e6

    def _llr_for_bit(self, llr_ch, u_partial, phi):
        """给定已判决前缀 u_partial[0:phi]，计算第 phi 位的 LLR"""
        N = self.N
        n = self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        for k in range(phi):
            R[k, 0] = self.large if u_partial[k] == 0 else -self.large
        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                L[i : i + s, j - 1] = f_operation(
                    R[i : i + s, j] + L[i + s : i + 2 * s, j],
                    L[i : i + s, j],
                )
                L[i + s : i + 2 * s, j - 1] = f_operation(
                    R[i : i + s, j], L[i : i + s, j]
                ) + L[i + s : i + 2 * s, j]
        for j in range(0, n):
            s = 1 << j
            for i in range(0, N, 2 * s):
                R[i : i + s, j + 1] = f_operation(
                    R[i + s : i + 2 * s, j] + L[i + s : i + 2 * s, j + 1],
                    R[i : i + s, j],
                )
                R[i + s : i + 2 * s, j + 1] = f_operation(
                    R[i : i + s, j], L[i : i + s, j + 1]
                ) + R[i + s : i + 2 * s, j]
        return L[phi, 0] + R[phi, 0]

    def _path_metric_penalty(self, llr, u):
        preferred = 0 if llr >= 0 else 1
        return abs(llr) if u != preferred else 0.0

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path()]
        paths[0].u_hat = np.zeros(self.N, dtype=np.int8)

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_bit = self._llr_for_bit(llr_ch, path.u_hat, phi)
                if self.frozen_bits[phi]:
                    child = _Path()
                    child.pm = path.pm + self._path_metric_penalty(llr_bit, 0)
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[phi] = 0
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = _Path()
                        child.pm = path.pm + self._path_metric_penalty(llr_bit, u)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[phi] = u
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                bits = path.u_hat[self.info_indices]
                if crc_check(bits, self.crc_length):
                    return path.u_hat.copy(), path.pm
        best = paths[0]
        return best.u_hat.copy(), best.pm
