"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)
def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 编码，附加校验位到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], recomputed)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _llr_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def _update_llrs(self, L, B, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l, bit):
        B[l, self.n] = bit
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L_size = self.N, self.n, self.list_size

        paths = []
        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_ch
        paths.append({"pm": 0.0, "L": L0, "B": B0, "u": np.zeros(N, dtype=np.int8)})

        for phi_natural in range(N):
            l = bit_reversed(phi_natural, n)
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, n]
                if np.isnan(llr):
                    llr = 0.0

                if l in self.frozen_set:
                    penalty = self._llr_penalty(llr, 0)
                    candidates.append((path["pm"] + penalty, pidx, 0))
                else:
                    for bit in (0, 1):
                        penalty = self._llr_penalty(llr, bit)
                        candidates.append((path["pm"] + penalty, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[:L_size]

            new_paths = []
            for pm, parent_idx, bit in selected:
                parent = paths[parent_idx]
                child = {
                    "pm": pm,
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "u": parent["u"].copy(),
                }
                child["u"][l] = bit
                self._update_bits(child["B"], l, bit)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = [i for i, p in enumerate(paths) if crc_check(p["u"], self.crc_length)]
            best = min(valid if valid else range(len(paths)), key=lambda i: paths[i]["pm"])
        else:
            best = min(range(len(paths)), key=lambda i: paths[i]["pm"])

        return paths[best]["u"].astype(int), paths[best]["pm"]
