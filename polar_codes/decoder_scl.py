"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, sc_decode
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_mod(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_mod(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _g_tuple(l1, l2, decision):
    bits = decision[1]
    return [l2[i] + (1 - 2 * bits[i]) * l1[i] for i in range(len(l2))]


def _xor_paths(left, right, left_list, right_list):
    res_bits = [(left[1][i] + right[1][i]) % 2 for i in range(len(left[1]))]
    res_bits.extend(right[1])
    return (left[0] + right[0], res_bits, left_list + right_list)


def _scl_tree_decode(y, depth, node, frozen_set, n, list_size):
    if depth == n - 1:
        decisions = []
        decoded_lists = []
        if node in frozen_set:
            pm = abs(y[0]) if y[0] < 0 else 0.0
            decisions.append((pm, [0]))
            decoded_lists.append([0])
        else:
            if y[0] < 0:
                decisions.append((0.0, [1]))
                decoded_lists.append([1])
                decisions.append((abs(y[0]), [0]))
                decoded_lists.append([0])
            else:
                decisions.append((0.0, [0]))
                decoded_lists.append([0])
                decisions.append((abs(y[0]), [1]))
                decoded_lists.append([1])
        return decisions, decoded_lists

    half = len(y) // 2
    l1, l2 = y[:half], y[half:]
    left_llr = f_operation(l1, l2).tolist()
    l_decisions, l_lists = _scl_tree_decode(
        left_llr, depth + 1, 2 * node, frozen_set, n, list_size
    )

    selection = []
    for i, l_dec in enumerate(l_decisions):
        right_llr = _g_tuple(l1, l2, l_dec)
        r_decisions, r_lists = _scl_tree_decode(
            right_llr, depth + 1, 2 * node + 1, frozen_set, n, list_size
        )
        for j, r_dec in enumerate(r_decisions):
            selection.append(_xor_paths(l_dec, r_dec, l_lists[i], r_lists[j]))

    selection.sort(key=lambda x: x[0])
    selection = selection[:list_size]
    return [(s[0], s[1]) for s in selection], [s[2] for s in selection]


class SCLDecoder:
    """SCL 译码器（因子树列表译码）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_set = set(np.where(self.frozen_bits.astype(bool))[0])

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self.br].tolist()
        decisions, decoded_lists = _scl_tree_decode(
            llr_perm, 0, 0, self.frozen_set, self.n, self.list_size
        )

        candidates = []
        for (pm, _), bit_list in zip(decisions, decoded_lists):
            u_hat = np.zeros(self.N, dtype=int)
            for i, bit in enumerate(bit_list):
                if i < self.N:
                    u_hat[i] = bit
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            valid = [
                (pm, u) for pm, u in candidates
                if crc_check(u[info_idx], self.crc_length)
            ]
            pool = valid if valid else candidates
        else:
            pool = candidates

        best = min(pool, key=lambda x: x[0])
        return best[1], best[0]
