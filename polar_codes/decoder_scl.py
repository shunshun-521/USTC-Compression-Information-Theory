"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（路径级 L/B 阵列）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = []
        L0 = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B0 = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch[self.br]
        paths.append({"L": L0, "B": B0, "pm": 0.0, "u": np.zeros(self.N, dtype=int)})

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                L, B, pm = path["L"], path["B"], path["pm"]
                for s in range(self.n - _active_llr_level(l, self.n), self.n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, self.N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                            )

                llr_bit = L[l, self.n]
                if l in self.frozen_set:
                    cand = [(pm + self._pm_penalty(llr_bit, 0), 0)]
                else:
                    cand = [
                        (pm + self._pm_penalty(llr_bit, 0), 0),
                        (pm + self._pm_penalty(llr_bit, 1), 1),
                    ]

                for new_pm, bit in cand:
                    Lc = L.copy()
                    Bc = B.copy()
                    uc = path["u"].copy()
                    uc[l] = bit
                    Bc[l, self.n] = bit
                    if l >= self.N / 2:
                        for s in range(
                            self.n, self.n - _active_bit_level(l, self.n), -1
                        ):
                            block_size = 1 << s
                            branch_size = block_size // 2
                            for j in range(l, -1, -block_size):
                                if j % block_size >= branch_size:
                                    Bc[j - branch_size, s - 1] = (
                                        Bc[j, s] ^ Bc[j - branch_size, s]
                                    )
                                    Bc[j, s - 1] = Bc[j, s]
                    new_paths.append({"L": Lc, "B": Bc, "pm": new_pm, "u": uc})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best_crc = None
        best_all = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            for p in paths:
                info_bits = p["u"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or p["pm"] < best_crc["pm"]:
                        best_crc = p
        chosen = best_crc if best_crc is not None else best_all
        return chosen["u"].copy(), chosen["pm"]
