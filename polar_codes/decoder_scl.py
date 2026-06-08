"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import active_bit_level, active_llr_level, bit_reversed_index, f_operation


def crc_encode(info_bits, crc_length=8):
    """CRC 编码：返回 [信息比特 | CRC 校验位]"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)

    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(8):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC（余数为零则通过）"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)

    for bit in bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(8):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=int),
                "u": np.zeros(self.N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phase_i in range(self.N):
            l = bit_reversed_index(phase_i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    p2 = self._clone_path(path)
                    if llr_bit < 0:
                        p2["pm"] += abs(llr_bit)
                    p2["u"][l] = 0
                    p2["B"][l, self.n] = 0
                    self._update_bits(p2, l)
                    new_paths.append(p2)
                else:
                    for bit in (0, 1):
                        p2 = self._clone_path(path)
                        if (bit == 0 and llr_bit < 0) or (bit == 1 and llr_bit >= 0):
                            p2["pm"] += abs(llr_bit)
                        p2["u"][l] = bit
                        p2["B"][l, self.n] = bit
                        self._update_bits(p2, l)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        return self._select_best(paths)

    def _clone_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u": path["u"].copy(),
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if top_bit == 0:
                        L[j, s + 1] = L[j, s] + L[j - branch_size, s]
                    else:
                        L[j, s + 1] = L[j, s] - L[j - branch_size, s]

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _select_best(self, paths):
        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p["u"])]
            pool = valid if valid else paths
        else:
            pool = paths
        best = min(pool, key=lambda p: p["pm"])
        return best["u"].astype(int), best["pm"]

    def _crc_valid(self, u):
        payload = u[self.info_indices]
        if len(payload) < self.crc_length:
            return False
        return crc_check(payload, self.crc_length)
