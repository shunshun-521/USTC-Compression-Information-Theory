"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _frozen_index_set,
    f_operation,
    g_operation,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    payload = np.concatenate([bits[:-crc_length], np.zeros(crc_length, dtype=int)])
    rem = _crc_remainder(payload, poly, crc_length)
    expected = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.array_equal(bits[-crc_length:], expected)


def _update_llrs(L, B, l, n):
    """为当前路径更新 LLR 树，得到 L[l, n]。"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（每条路径独立维护 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_index_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        def new_path():
            L = np.zeros((N, n + 1), dtype=np.float64)
            L[:, 0] = llr_ch
            return {
                "pm": 0.0,
                "B": np.zeros((N, n + 1), dtype=int),
                "u": np.zeros(N, dtype=int),
                "L": L,
            }

        paths = [new_path()]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for pidx, path in enumerate(paths):
                _update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]
                if l in self.frozen_set:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    candidates.append((path["pm"] + penalty, pidx, 0))
                else:
                    pm0 = path["pm"] + (0.0 if llr >= 0 else abs(llr))
                    pm1 = path["pm"] + (0.0 if llr < 0 else abs(llr))
                    candidates.append((pm0, pidx, 0))
                    candidates.append((pm1, pidx, 1))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent_idx, bit in candidates:
                parent = paths[parent_idx]
                child = {
                    "pm": pm,
                    "B": parent["B"].copy(),
                    "u": parent["u"].copy(),
                    "L": parent["L"].copy(),
                }
                child["B"][l, n] = bit
                child["u"][l] = bit
                _update_bits(child["B"], l, n, N)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            info_idx = np.array(sorted(set(range(self.N)) - self.frozen_set), dtype=int)
            crc_pass = [
                i
                for i, p in enumerate(paths)
                if crc_check(p["u"][info_idx], self.crc_length)
            ]
            if crc_pass:
                best = min(crc_pass, key=lambda i: paths[i]["pm"])
            else:
                best = min(range(len(paths)), key=lambda i: paths[i]["pm"])
        else:
            best = min(range(len(paths)), key=lambda i: paths[i]["pm"])

        return paths[best]["u"], paths[best]["pm"]
