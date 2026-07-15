"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _frozen_phase_set,
    f_boxplus,
    g_operation,
)
from encoder import bit_reversal_permutation

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 B，LLR 树共享信道输入）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_phase = _frozen_phase_set(self.frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_idx = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def _update_llrs(self, L, B, phase):
        l = _bit_reversed(phase, self.n)
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )
        return L[l, self.n]

    def _update_bits(self, B, phase):
        l = _bit_reversed(phase, self.n)
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
                  "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64)}]
        paths[0]["L"][:, 0] = llr_ch[self.br]

        for phase in range(self.N):
            expanded = []
            for path in paths:
                L = path["L"]
                llr_bit = self._update_llrs(L, path["B"], phase)
                l = _bit_reversed(phase, self.n)

                if phase in self.frozen_phase:
                    new_b = path["B"].copy()
                    new_b[l, self.n] = 0
                    self._update_bits(new_b, phase)
                    expanded.append(
                        {
                            "pm": path["pm"] + self._penalty(llr_bit, 0),
                            "B": new_b,
                            "L": L,
                        }
                    )
                else:
                    for bit in (0, 1):
                        new_b = path["B"].copy()
                        new_b[l, self.n] = bit
                        self._update_bits(new_b, phase)
                        expanded.append(
                            {
                                "pm": path["pm"] + self._penalty(llr_bit, bit),
                                "B": new_b,
                                "L": copy.deepcopy(L),
                            }
                        )

            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = path["B"][:, self.n].astype(int)
            pm = path["pm"]
            if self.crc_length > 0:
                payload = u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    candidates.append((pm, u_hat))
            else:
                candidates.append((pm, u_hat))

        if not candidates:
            candidates = [(p["pm"], p["B"][:, self.n].astype(int)) for p in paths]

        best_pm, best_u = min(candidates, key=lambda x: x[0])
        return best_u, best_pm
