"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0].tolist())
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _new_path(self, llr_ch):
        return {
            "pm": 0.0,
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "u_hat": np.zeros(self.N, dtype=int),
        }

    @staticmethod
    def _branch_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _update_llrs(self, state, l):
        L = state["L"]
        B = state["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, state, l):
        B = state["B"]
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []
            for state in paths:
                self._update_llrs(state, l)
                llr_dec = state["L"][l, self.n]
                if l in self.frozen_set:
                    ns = {
                        "pm": state["pm"] + self._branch_penalty(llr_dec, 0),
                        "L": state["L"].copy(),
                        "B": state["B"].copy(),
                        "u_hat": state["u_hat"].copy(),
                    }
                    ns["B"][l, self.n] = 0
                    ns["u_hat"][l] = 0
                    candidates.append(ns)
                else:
                    for u in (0, 1):
                        ns = {
                            "pm": state["pm"] + self._branch_penalty(llr_dec, u),
                            "L": state["L"].copy(),
                            "B": state["B"].copy(),
                            "u_hat": state["u_hat"].copy(),
                        }
                        ns["B"][l, self.n] = u
                        ns["u_hat"][l] = u
                        candidates.append(ns)

            candidates.sort(key=lambda s: s["pm"])
            paths = candidates[: self.list_size]
            for state in paths:
                self._update_bits(state, l)

        paths.sort(key=lambda s: s["pm"])
        if self.crc_length > 0:
            for state in paths:
                bits = state["u_hat"][self.info_indices]
                if crc_check(bits, self.crc_length):
                    return state["u_hat"].copy(), state["pm"]
        best = paths[0]
        return best["u_hat"].copy(), best["pm"]
