"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _reorder_channel_llrs,
    f_operation,
    g_operation,
    path_metric_penalty,
)


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]  # x^8 + x^2 + x + 1
CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # CRC-16-IBM


def _crc_poly(crc_length):
    return CRC8_POLY if crc_length == 8 else CRC16_POLY


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    msg = list(map(int, bits))
    while len(msg) > crc_length:
        if msg[0]:
            for i in range(len(poly)):
                msg[i] ^= poly[i]
        msg.pop(0)
    return np.array(msg[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), crc_length
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    rem = _crc_remainder(bits, crc_length)
    return np.all(rem == 0)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
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

    def _update_bits(self, path, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = _reorder_channel_llrs(llr_ch)
        N, n = self.N, self.n
        L_size = self.list_size

        paths = [_Path(n, N) for _ in range(L_size)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed_index(phi, n)
            candidates = []

            for path in paths[: min(len(paths), L_size)]:
                self._update_llrs(path, l)
                llr = path.L[l, n]
                if self.frozen_bits[l]:
                    candidates.append((path.pm + path_metric_penalty(llr, 0), path, 0))
                else:
                    for u in (0, 1):
                        candidates.append(
                            (path.pm + path_metric_penalty(llr, u), path, u)
                        )

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm_new, parent, u in candidates:
                if len(new_paths) >= L_size:
                    break
                child = _Path(n, N)
                child.pm = pm_new
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.u_hat[l] = u
                child.B[l, n] = u
                self._update_bits(child, l)
                new_paths.append(child)

            paths = new_paths[:L_size]

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
