"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _bit_reversed_index, _update_llrs, _update_bits


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly, mask = 0x07, 0xFF
    elif crc_length == 16:
        poly, mask = 0x8005, 0xFFFF
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, llr, n, N):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径复制 + PM 剪枝）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_add(pm, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        if u_bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        rev = bit_reversal_permutation(self.N)
        llr = np.asarray(llr_ch, dtype=np.float64)[rev]
        N, n = self.N, self.n

        paths = [_Path(llr, n, N)]

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []

            for p_idx, path in enumerate(paths):
                _update_llrs(path.L, path.B, l, n, N)
                cur_llr = path.L[l, n]
                if self.frozen_bits[l]:
                    pm = self._pm_add(path.pm, cur_llr, 0)
                    candidates.append((pm, p_idx, 0))
                else:
                    for u_bit in (0, 1):
                        pm = self._pm_add(path.pm, cur_llr, u_bit)
                        candidates.append((pm, p_idx, u_bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent_idx, u_bit in candidates:
                parent = paths[parent_idx]
                child = _Path(llr, n, N)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat = parent.u_hat.copy()
                child.u_hat[l] = u_bit
                child.B[l, n] = u_bit
                _update_bits(child.B, l, n, N)
                new_paths.append(child)
            paths = new_paths

        decoded_paths = [(p.pm, p.u_hat.copy()) for p in paths]
        best_pm, best_u = min(decoded_paths, key=lambda x: x[0])
        best_u[self.frozen_bits] = 0

        if self.crc_length > 0:
            valid = []
            for pm, u_hat in decoded_paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                best_pm, best_u = min(valid, key=lambda x: x[0])

        return best_u, best_pm
