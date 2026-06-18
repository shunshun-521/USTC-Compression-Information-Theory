"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import _sc_matrix_decode, _frozen_to_info_indices
import sc_core as scf


def _crc_poly(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError(f'Unsupported CRC length: {crc_length}')
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = [int(b) for b in np.asarray(info_bits, dtype=int)]
    p = _crc_poly(crc_length)
    work = info_bits + [0] * crc_length
    for i in range(len(info_bits)):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[i + j] ^= p[j]
    return np.array(info_bits + work[-crc_length:], dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = [int(b) for b in np.asarray(bits, dtype=int)]
    return crc_encode(bits[:-crc_length], crc_length).tolist() == bits


def _sc_step_to_split(llr_matrix, bit_matrix, information_pos, split_pos, frozen_bit=0):
    """译码至 split_pos 位判决完成。"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = scf.get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if scf.all_num(up_bit) == 1:
            position = scf.up(position)
        elif scf.all_num(right_bit) == 1:
            bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = scf.get_up_bit(left_bit, right_bit)
        elif scf.all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                rb = scf.get_right_bit(right_llr, information_pos, frozen_bit, position[1] + 1)
                bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = rb
            else:
                position = scf.rightdown(position)
        elif scf.all_num(left_bit) == 1:
            llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = scf.get_right_llr(left_bit, up_llr)
        elif scf.all_num(left_llr) == 0:
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = scf.get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                lb = scf.get_left_bit(left_llr, information_pos, frozen_bit, position[1])
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = lb
            else:
                position = scf.leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.info_indices = _frozen_to_info_indices(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_state(self, llr_ch):
        br = bit_reversal_permutation(self.N)
        llr = np.asarray(llr_ch, dtype=np.float64)[br]
        llr_matrix = np.ones((self.n + 1, self.N))
        llr_matrix[llr_matrix == 1] = float('nan')
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = _sc_matrix_decode(
                np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(self.N)],
                self.info_indices,
                0,
            )
            return u_hat, 0.0

        n = self.n
        split_pos = list(self.info_indices)
        llr_list = []
        bit_list = []
        pm_list = [0.0]
        llr_m, bit_m = self._init_state(llr_ch)
        llr_list.append(llr_m)
        bit_list.append(bit_m)
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            new_llr, new_bit, new_pm = [], [], []
            for i in range(l_now):
                lc, bc = llr_list[i].copy(), bit_list[i].copy()
                lc, bc = _sc_step_to_split(lc, bc, self.info_indices, split_pos[split_loc])
                prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                seg_llr = lc[n][prev:split_pos[split_loc] + 1]
                seg_bit = bc[n][prev:split_pos[split_loc] + 1]
                pm_base = pm_list[i] + scf.get_pm_update(seg_llr, seg_bit, 'hf')

                new_llr.append(lc)
                new_bit.append(bc)
                new_pm.append(pm_base)

                bc_wrong = bc.copy()
                bc_wrong[n][split_pos[split_loc]] = 1 - bc_wrong[n][split_pos[split_loc]]
                seg_bit_w = bc_wrong[n][prev:split_pos[split_loc] + 1]
                pm_wrong = pm_list[i] + scf.get_pm_update(seg_llr, seg_bit_w, 'hf')
                new_llr.append(lc.copy())
                new_bit.append(bc_wrong)
                new_pm.append(pm_wrong)

            order = np.argsort(new_pm)[:self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos[-1] != self.N - 1:
            for i in range(l_now):
                llr_list[i], bit_list[i] = _sc_step_to_split(
                    llr_list[i], bit_list[i], self.info_indices, self.N - 1
                )

        order = np.argsort(pm_list)
        if self.crc_length > 0:
            for idx in order:
                u_hat = bit_list[idx][n].astype(int)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    return u_hat, pm_list[idx]

        best = order[0]
        return bit_list[best][n].astype(int), pm_list[best]
