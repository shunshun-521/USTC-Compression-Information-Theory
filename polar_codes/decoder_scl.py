"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _bit_reversed_index,
    _channel_llr_layout,
    _permuted_scd,
    active_bit_level,
    active_llr_level,
    lower_llr,
    sc_decode,
    upper_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length <= 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return bool(np.array_equal(expected, bits))


class SCLDecoder:
    """SCL 译码器（Permuted SCD 列表扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _init_path(self, llr_layout):
        return {
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
            "llr_layout": llr_layout,
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path["B"]
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_layout = _channel_llr_layout(llr_ch)
        paths = [self._init_path(llr_layout)]
        paths[0]["L"][:, 0] = llr_layout

        for l in [_bit_reversed_index(i, self.n) for i in range(self.N)]:
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]
                if l in self.frozen_set:
                    pm = path["pm"] + self._metric_penalty(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._metric_penalty(llr, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in candidates:
                child = self._init_path(llr_layout)
                child["L"] = parent["L"].copy()
                child["B"] = parent["B"].copy()
                child["pm"] = pm
                child["u_hat"] = parent["u_hat"].copy()
                child["u_hat"][l] = bit
                child["B"][l, self.n] = bit
                child["L"][:, 0] = llr_layout
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
