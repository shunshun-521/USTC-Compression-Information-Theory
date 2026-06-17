"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _pm_penalty(llr, u):
    u_from_llr = 0 if llr >= 0 else 1
    return 0.0 if u == u_from_llr else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        _, self.llr_layer_vec, self.bit_layer_vec, self.decode_order = precompute_sc_indices(N)

    def _update_llrs(self, path, idx, l):
        L = path["L"]
        B = path["B"]
        for s in self.llr_layer_vec[idx]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, idx, l):
        B = path["B"]
        for s in self.bit_layer_vec[idx]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _new_path(self, template=None):
        if template is None:
            path = {
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
                "pm": 0.0,
                "u_hat": np.zeros(self.N, dtype=int),
            }
            return path
        return {
            "L": template["L"].copy(),
            "B": template["B"].copy(),
            "pm": template["pm"],
            "u_hat": template["u_hat"].copy(),
        }

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [self._new_path()]
        paths[0]["L"][:, 0] = llr_ch

        for idx, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                self._update_llrs(path, idx, l)
                leaf_llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = self._new_path(path)
                    new_path["pm"] += _pm_penalty(leaf_llr, 0)
                    new_path["u_hat"][l] = 0
                    new_path["B"][l, self.n] = 0
                    self._update_bits(new_path, idx, l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._new_path(path)
                        new_path["pm"] += _pm_penalty(leaf_llr, u)
                        new_path["u_hat"][l] = u
                        new_path["B"][l, self.n] = u
                        self._update_bits(new_path, idx, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            chosen = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            chosen = min(paths, key=lambda p: p["pm"])

        return chosen["u_hat"].copy(), chosen["pm"]
