"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x107
CRC16_POLY = 0x8005


def _crc_division(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    remainder = _crc_division(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class Path:
    __slots__ = ("pm", "B", "L", "active")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.rev = bit_reversal_permutation(N)

    def _copy_path(self, src, dst):
        dst.pm = src.pm
        dst.L[:] = src.L
        dst.B[:] = src.B
        dst.active = True

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

    def _update_bits(self, path, l):
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

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = [Path(N, n) for _ in range(L_size)]
        paths[0].L[:, 0] = llr_ch[self.rev]
        num_active = 1

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for p_idx in range(num_active):
                path = paths[p_idx]
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    candidates.append((path.pm + self._pm_penalty(llr, 0), p_idx, 0))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (path.pm + self._pm_penalty(llr, bit), p_idx, bit)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L_size]

            new_paths = [Path(N, n) for _ in range(L_size)]
            for idx, (pm, src_idx, bit) in enumerate(candidates):
                self._copy_path(paths[src_idx], new_paths[idx])
                new_paths[idx].pm = pm
                new_paths[idx].B[l, n] = bit
                self._update_bits(new_paths[idx], l)

            paths = new_paths
            num_active = len(candidates)

        best_crc = None
        if self.crc_length > 0:
            for path in paths[:num_active]:
                u_hat = path.B[:, n].astype(int)
                if crc_check(u_hat, self.crc_length):
                    if best_crc is None or path.pm < best_crc[0]:
                        best_crc = (path.pm, path)

        chosen = (
            best_crc[1]
            if best_crc is not None
            else min(paths[:num_active], key=lambda p: p.pm)
        )
        return chosen.B[:, n].astype(int).copy(), chosen.pm
