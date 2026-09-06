"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
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
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _encoded_partial(prefix, offset, length):
    """从已知前缀构造子树的 u_hat_up（用于 g 运算）。"""
    if length == 1:
        b = prefix[offset]
        return np.array([b], dtype=int), np.array([b], dtype=int)
    half = length // 2
    u1, u1_up = _encoded_partial(prefix, offset, half)
    u2, u2_up = _encoded_partial(prefix, offset + half, half)
    u_up = np.concatenate([(u1_up ^ u2_up).astype(int), u2_up.astype(int)])
    return np.concatenate([u1, u2]), u_up


def _llr_at_phi(llr, frozen_bits, phi, prefix):
    """在已知 prefix（索引 < phi）时，返回第 phi 位的 LLR。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool).astype(int)
    N = len(llr)

    def walk(llr_node, frozen_node, offset, length):
        if length == 1:
            return float(llr_node[0])
        half = length // 2
        if phi < offset + half:
            left_llr = f_operation(llr_node[:half], llr_node[half:])
            return walk(left_llr, frozen_node[:half], offset, half)
        _, u_left_up = _encoded_partial(prefix, offset, half)
        right_llr = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        return walk(right_llr, frozen_node[half:], offset + half, half)

    return walk(llr, frozen_bits, 0, N)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = llr_ch[self.br].astype(np.float64)
        paths = [{"pm": 0.0, "prefix": {}}]

        for phi in range(self.N):
            candidates = []
            for state in paths:
                llr_phi = _llr_at_phi(llr, self.frozen_bits, phi, state["prefix"])

                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr_phi >= 0 else abs(llr_phi)
                    child = {"pm": state["pm"] + penalty, "prefix": dict(state["prefix"])}
                    child["prefix"][phi] = 0
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        hard = 0 if llr_phi >= 0 else 1
                        penalty = 0.0 if bit == hard else abs(llr_phi)
                        child = {
                            "pm": state["pm"] + penalty,
                            "prefix": dict(state["prefix"]),
                        }
                        child["prefix"][phi] = bit
                        candidates.append(child)

            candidates.sort(key=lambda s: s["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for state in paths:
                u_hat = self._assemble(state["prefix"])
                if crc_check(u_hat, self.crc_length):
                    valid.append(state)
            best = min(valid if valid else paths, key=lambda s: s["pm"])
        else:
            best = min(paths, key=lambda s: s["pm"])

        return self._assemble(best["prefix"]), best["pm"]

    def _assemble(self, prefix):
        u_hat = np.zeros(self.N, dtype=int)
        for idx, bit in prefix.items():
            u_hat[idx] = bit
        return u_hat
