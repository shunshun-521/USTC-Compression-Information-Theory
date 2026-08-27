"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)
from encoder import channel_llr_to_decoder

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class Path:
    """SCL 译码单条路径"""

    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _continue_path(self, path, l, u_val):
        new_path = Path(self.N, self.n, path.L[:, 0])
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()

        n = self.n
        N = self.N

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    new_path.L[j, s + 1] = f_operation(
                        new_path.L[j, s], new_path.L[j + branch_size, s]
                    )
                else:
                    new_path.L[j, s + 1] = g_operation(
                        new_path.L[j - branch_size, s],
                        new_path.L[j, s],
                        new_path.B[j - branch_size, s + 1],
                    )

        new_path.B[l, n] = u_val
        new_path.u_hat[l] = u_val

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        new_path.B[j - branch_size, s - 1] = (
                            new_path.B[j, s] ^ new_path.B[j - branch_size, s]
                        )
                        new_path.B[j, s - 1] = new_path.B[j, s]

        return new_path

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_internal = channel_llr_to_decoder(llr_ch)
        N = self.N
        n = self.n
        decode_order = [_bit_reversed(i, n) for i in range(N)]

        paths = [Path(N, n, llr_internal)]

        for l in decode_order:
            candidates = []
            for path in paths:
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
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

                llr = path.L[l, n]

                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(llr, 0)
                    new_path = self._continue_path(path, l, 0)
                    new_path.pm += penalty
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        penalty = self._path_metric_penalty(llr, u)
                        new_path = self._continue_path(path, l, u)
                        new_path.pm += penalty
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_pass = []
        for p in paths:
            info_bits = p.u_hat[self.info_indices]
            if self.crc_length > 0:
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        best = min(crc_pass or paths, key=lambda p: p.pm)
        return best.u_hat, best.pm
