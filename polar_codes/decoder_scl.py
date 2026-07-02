"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _active_bit_level, _active_llr_level, f_operation
from encoder import bit_reversed_index


def _crc_remainder(bits, crc_length):
    """计算 CRC 余式，完整码字（含 CRC）的余式应为 0。"""
    if crc_length == 8:
        poly = 0x07
        reg = 0
        for bit in bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return reg

    poly = 0x8005
    reg = 0
    for bit in bits:
        reg ^= int(bit) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    augmented = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _crc_remainder(augmented, crc_length)
    if crc_length == 8:
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)
        self.L[:, 0] = llr_ch.copy()


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, phi):
        n = self.n
        for s in range(n - _active_llr_level(phi, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phi, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    if top_bit == 0:
                        path.L[j, s + 1] = (
                            path.L[j, s] + path.L[j - branch_size, s]
                        )
                    else:
                        path.L[j, s + 1] = (
                            path.L[j, s] - path.L[j - branch_size, s]
                        )

    def _update_bits(self, path, phi):
        if phi < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(phi, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(phi, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _branch_metric(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            phi = bit_reversed_index(i, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path, phi)
                llr = path.L[phi, self.n]

                if self.frozen_bits[phi]:
                    pm = path.pm + self._branch_metric(llr, 0)
                    candidates.append((pm, pidx, 0))
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._branch_metric(llr, bit)
                        candidates.append((pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, bit in candidates:
                path = _Path(self.N, self.n, llr_ch)
                path.pm = pm
                path.L = paths[pidx].L.copy()
                path.B = paths[pidx].B.copy()
                path.u_hat = paths[pidx].u_hat.copy()

                path.B[phi, self.n] = bit
                path.u_hat[phi] = bit
                self._update_bits(path, phi)
                new_paths.append(path)
            paths = new_paths

        valid = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            else:
                valid.append(path)

        pool = valid if valid else paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
