"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import importlib.util
import math
import os
import numpy as np
from decoder_sc import _permute_llr_for_decode, _frozen_to_info_pos, sc_decode_nonrecursive

_REF_FUNCTION_PATH = os.path.join(os.path.dirname(__file__), '_ref_function.py')
_spec = importlib.util.spec_from_file_location('polar_ref_function', _REF_FUNCTION_PATH)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = ((reg << 1) & ((1 << crc_length) - 1))
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    actual = 0
    for b in bits[-crc_length:]:
        actual = (actual << 1) | int(b)
    return remainder == actual


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _ref.get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if _ref.all_num(bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]) == 1:
            position = _ref.up(position)
        else:
            if _ref.all_num(right_bit) == 1:
                up_bit = _ref.get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
            elif _ref.all_num(right_llr) == 1:
                if position[0] == position[2] - 1:
                    right_bit = _ref.get_right_bit(right_llr, information_pos, frozen_bit, position[1] + 1)
                    bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_bit
                else:
                    position = _ref.rightdown(position)
            elif _ref.all_num(left_bit) == 1:
                right_llr = _ref.get_right_llr(left_bit, up_llr)
                llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_llr
            elif _ref.all_num(left_llr) == 0:
                left_llr = _ref.get_left_llr(up_llr)
                llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
            elif position[0] == position[2] - 1:
                left_bit = _ref.get_left_bit(left_llr, information_pos, frozen_bit, position[1])
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
            else:
                position = _ref.leftdown(position)

    return llr_matrix, bit_matrix


def _scl_decode_core(y_llr, information_pos, list_size, crc_length=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr

    llr_list = [llr_matrix.copy()]
    bit_list = [bit_matrix.copy()]
    pm_list = [0.0]
    split_pos = list(information_pos)
    split_loc = 0
    l_now = 1

    while split_loc < len(split_pos):
        for i in range(l_now):
            lm, bm = llr_list[i].copy(), bit_list[i].copy()
            pm = pm_list[i]
            lm, bm = _sc_stepping_decoder(lm, bm, information_pos, 0, split_pos[split_loc])
            llr_list[i] = lm
            bit_list[i] = bm

            prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
            cur = split_pos[split_loc] + 1
            pm_list[i] = pm + _ref.get_pm_update(lm[n, prev:cur], bm[n, prev:cur], 'hf')

            lm2, bm2 = lm.copy(), bm.copy()
            bm2[n, split_pos[split_loc]] = 1 - bm2[n, split_pos[split_loc]]
            llr_list.append(lm2)
            bit_list.append(bm2)
            pm_list.append(pm + _ref.get_pm_update(lm[n, prev:cur], bm2[n, prev:cur], 'hf'))

        order = np.argsort(pm_list)[:list_size]
        llr_list = [llr_list[i] for i in order]
        bit_list = [bit_list[i] for i in order]
        pm_list = [pm_list[i] for i in order]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos and split_pos[-1] != N - 1:
        for i in range(l_now):
            lm, bm = llr_list[i].copy(), bit_list[i].copy()
            lm, bm = _sc_stepping_decoder(lm, bm, information_pos, 0, N - 1)
            llr_list[i], bit_list[i] = lm, bm
            prev = split_pos[-1] + 1
            pm_list[i] += _ref.get_pm_update(lm[n, prev:N], bm[n, prev:N], 'hf')

    best_idx = int(np.argmin(pm_list))
    if crc_length > 0:
        for idx in np.argsort(pm_list):
            if crc_check(bit_list[idx][n].astype(int), crc_length):
                best_idx = idx
                break

    return bit_list[best_idx][n].astype(int), pm_list[best_idx]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_pos = _frozen_to_info_pos(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = _permute_llr_for_decode(llr_ch)
        if self.list_size == 1:
            u_hat = sc_decode_nonrecursive(llr_ch, self.info_pos, frozen_bit=0)
            return u_hat, 0.0
        return _scl_decode_core(llr_ch, self.info_pos, self.list_size, self.crc_length)


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=8.0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    return True
