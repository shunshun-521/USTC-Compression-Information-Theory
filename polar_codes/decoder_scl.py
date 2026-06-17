"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _prepare_channel_llr,
    f_operation,
    g_operation,
    sc_decode,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


def _path_metric_penalty(llr, bit):
    """与 SC 路径度量一致：硬判与 LLR 符号不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr0 = _prepare_channel_llr(llr_ch)
        n = self.n
        N = self.N

        paths = [
            {
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=int),
                "pm": 0.0,
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr0

        for phi in range(N):
            l = _bit_reversed_index(phi, n)
            candidates = []

            for path in paths:
                L, B = path["L"], path["B"]
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size >> 1
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                            )

                cur_llr = L[l, n]
                if self.frozen_bits[l]:
                    pm = path["pm"] + _path_metric_penalty(cur_llr, 0)
                    new_path = {
                        "L": L.copy(),
                        "B": B.copy(),
                        "pm": pm,
                        "u": path["u"].copy(),
                    }
                    new_path["u"][l] = 0
                    new_path["B"][l, n] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + _path_metric_penalty(cur_llr, bit)
                        new_path = {
                            "L": L.copy(),
                            "B": B.copy(),
                            "pm": pm,
                            "u": path["u"].copy(),
                        }
                        new_path["u"][l] = bit
                        new_path["B"][l, n] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

            for path in paths:
                B = path["B"]
                if l < N // 2:
                    continue
                for s in range(n, n - _active_bit_level(l, n), -1):
                    block_size = 1 << s
                    branch_size = block_size >> 1
                    for j in range(l, -1, -block_size):
                        if j % block_size >= branch_size:
                            B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                                B[j - branch_size, s]
                            )
                            B[j, s - 1] = B[j, s]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
