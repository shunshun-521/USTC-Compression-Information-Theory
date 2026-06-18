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
    check_code = work[-crc_length:]
    return np.array(info_bits + check_code, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = [int(b) for b in np.asarray(bits, dtype=int)]
    info = bits[:-crc_length]
    return crc_encode(info, crc_length).tolist() == bits


def _get_up_loc(bit_matrix):
    n = int(np.log2(bit_matrix.shape[1]))
    N = bit_matrix.shape[1]
    for i in range(N):
        if np.isnan(bit_matrix[n][i]):
            for layer in range(n + 1):
                width = 2 ** (n - layer)
                if i % width == 0:
                    return [layer, i]
    return [0, 0]


def _sc_step_to_split(llr_matrix, bit_matrix, information_pos, split_pos, frozen_bit=0):
    """译码至信息位 split_pos 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while not (bit_matrix[n][split_pos] == 0 or bit_matrix[n][split_pos] == 1):
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
        self.frozen_bits = np.asarray(frozen_bits)
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
        llr_list, bit_list, pm_list = [], [], []
        llr_m, bit_m = self._init_state(llr_ch)
        llr_list.append(llr_m)
        bit_list.append(bit_m)
        pm_list.append(0.0)

        for loc, split in enumerate(split_pos):
            prev = split_pos[loc - 1] if loc > 0 else -1
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                lc, bc = llr_m.copy(), bit_m.copy()
                lc, bc = _sc_step_to_split(lc, bc, self.info_indices, split)
                seg_llr = lc[n][prev + 1:split + 1]
                seg_bit = bc[n][prev + 1:split + 1]
                base_pm = pm + scf.get_pm_update(seg_llr, seg_bit, 'l')

                u_bit = int(bc[n][split])
                for val in (u_bit, 1 - u_bit):
                    lc2, bc2 = lc.copy(), bc.copy()
                    bc2[n][split] = val
                    pm_inc = 0.0 if val == u_bit else abs(float(lc[n][split]))
                    new_llr.append(lc2)
                    new_bit.append(bc2)
                    new_pm.append(base_pm + pm_inc)

            order = np.argsort(new_pm)[:self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]

        if split_pos[-1] != self.N - 1:
            for i in range(len(llr_list)):
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
