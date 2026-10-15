"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation, bit_reversed_index
from decoder_sc import (
    f_operation,
    _active_llr_level,
    _active_bit_level,
    _lower_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（路径复制 + PM 剪枝）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_add(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        return pm + (0.0 if hard == bit else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        N, n = self.N, self.n

        paths = [
            {
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "pm": 0.0,
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for out_i in range(N):
            l = bit_reversed_index(out_i, n)
            for p in paths:
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            p["L"][j, s + 1] = f_operation(
                                p["L"][j, s], p["L"][j + branch_size, s]
                            )
                        else:
                            p["L"][j, s + 1] = _lower_llr(
                                p["L"][j, s],
                                p["L"][j - branch_size, s],
                                int(p["B"][j - branch_size, s + 1]),
                            )

            llr_leaf = paths[0]["L"][l, n]  # 所有路径该位 LLR 相同
            # 实际上各路径 L 可能不同 due to B - recompute per path above

            if l in self.frozen_set:
                new_paths = []
                for p in paths:
                    cp = {
                        "L": p["L"].copy(),
                        "B": p["B"].copy(),
                        "pm": self._pm_add(p["pm"], p["L"][l, n], 0),
                        "u": p["u"].copy(),
                    }
                    cp["B"][l, n] = 0
                    cp["u"][l] = 0
                    self._update_bits(cp, l, n)
                    new_paths.append(cp)
                paths = new_paths
            else:
                candidates = []
                for p in paths:
                    for bit in (0, 1):
                        cp = {
                            "L": p["L"].copy(),
                            "B": p["B"].copy(),
                            "pm": self._pm_add(p["pm"], p["L"][l, n], bit),
                            "u": p["u"].copy(),
                        }
                        cp["B"][l, n] = bit
                        cp["u"][l] = bit
                        self._update_bits(cp, l, n)
                        candidates.append(cp)
                candidates.sort(key=lambda x: x["pm"])
                paths = candidates[: self.list_size]

        paths.sort(key=lambda x: x["pm"])
        if self.crc_length > 0:
            for p in paths:
                info_bits = p["u"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return p["u"], p["pm"]
        best = paths[0]
        return best["u"], best["pm"]

    @staticmethod
    def _update_bits(p, l, n):
        if l < (1 << (n - 1)):
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    p["B"][j - branch_size, s - 1] = int(p["B"][j, s]) ^ int(
                        p["B"][j - branch_size, s]
                    )
                    p["B"][j, s - 1] = p["B"][j, s]
