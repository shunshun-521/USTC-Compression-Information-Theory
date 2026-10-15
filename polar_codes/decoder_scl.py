"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _upper_llr,
    _lower_llr,
    _active_llr_level,
    _active_bit_level,
    _frozen_to_set,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加到信息比特后（MSB first）"""
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC"""
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器（每条路径独立 L/B，Lazy Copy 通过复制 B 与 u 实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen = _frozen_to_set(frozen_bits)
        self.br = bit_reversal_permutation(N)
        if info_indices is None:
            info_indices = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
        self.info_indices = np.asarray(info_indices, dtype=int)

    def _pm_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def _update_llrs(self, l, L, B):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, l, B):
        n, N = self.n, self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        N, n = self.N, self.n
        base_llr = llr_ch[self.br].astype(np.float64)

        def new_path():
            L = np.zeros((N, n + 1), dtype=np.float64)
            L[:, 0] = base_llr
            return {"L": L, "B": np.zeros((N, n + 1)), "pm": 0.0, "u": np.zeros(N, dtype=int)}

        paths = [new_path()]

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []
            for p in paths:
                self._update_llrs(l, p["L"], p["B"])
                llr = p["L"][l, n]
                if l in self.frozen:
                    B = p["B"].copy()
                    pm = self._pm_update(p["pm"], llr, 0)
                    B[l, n] = 0
                    self._update_bits(l, B)
                    u = p["u"].copy()
                    u[l] = 0
                    L = p["L"].copy()
                    new_paths.append({"L": L, "B": B, "pm": pm, "u": u})
                else:
                    for bit in (0, 1):
                        B = p["B"].copy()
                        pm = self._pm_update(p["pm"], llr, bit)
                        B[l, n] = bit
                        self._update_bits(l, B)
                        u = p["u"].copy()
                        u[l] = bit
                        L = p["L"].copy()
                        new_paths.append({"L": L, "B": B, "pm": pm, "u": u})
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                seq = p["u"][self.info_indices]
                if crc_check(seq, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x["pm"])
        return best["u"], best["pm"]
