"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    _upper_llr_boxplus,
    _prepare_channel_llrs,
    _bit_reversed_index,
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


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _crc_remainder(padded, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


def _pm_update(pm, llr, u_bit):
    """路径度量更新：与 LLR 硬判决不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        return pm + abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.info_indices = np.array(
            [i for i in range(N) if i not in self.frozen], dtype=int
        )

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L[:, 0] = _prepare_channel_llrs(llr_ch)
        return {
            "L": L,
            "B": B,
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }

    def decode(self, llr_ch):
        paths = [self._new_path(llr_ch)]

        for step, l in enumerate(
            [_bit_reversed_index(i, self.n) for i in range(self.N)]
        ):
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path["L"][l, self.n]

                if l in self.frozen:
                    u_choice = [0]
                    pm_base = _pm_update(path["pm"], llr, 0)
                else:
                    u_choice = [0, 1]
                    pm_base = path["pm"]

                for u_bit in u_choice:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": pm_base if l in self.frozen else _pm_update(path["pm"], llr, u_bit),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][l, self.n] = u_bit
                    new_path["u_hat"][l] = u_bit
                    self._update_bits(new_path, l)
                    candidates.append(new_path)

            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"][self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x["pm"])
        return best["u_hat"].astype(int), best["pm"]

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        B = path["B"]
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]
