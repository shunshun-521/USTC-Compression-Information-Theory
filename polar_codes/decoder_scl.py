"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    precompute_sc_indices,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


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
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = CRC_POLYS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int32,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int32)
    poly = CRC_POLYS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        precompute_sc_indices(N)

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int32),
            "u_hat": np.zeros(N, dtype=np.int32),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n, N)
                llr = L[l, n]

                if self.frozen_bits[i]:
                    child = {
                        "pm": path["pm"] + self._pm_penalty(llr, 0),
                        "L": L.copy(),
                        "B": B.copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["B"][l, n] = 0
                    child["u_hat"][i] = 0
                    _update_bits(child["B"], l, n, N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = {
                            "pm": path["pm"] + self._pm_penalty(llr, u),
                            "L": L.copy(),
                            "B": B.copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["B"][l, n] = u
                        child["u_hat"][i] = u
                        _update_bits(child["B"], l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u_hat"][~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p["pm"]) if valid else paths[0]
        else:
            best = paths[0]

        return best["u_hat"].copy(), best["pm"]
