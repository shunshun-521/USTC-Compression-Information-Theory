"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _all_defined,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _frozen_to_info_pos,
)


_CRC_POLY_LOCS = {
    8: [8, 2, 1, 0],
    16: [16, 15, 2, 0],
}


def _crc_poly(crc_length):
    locs = _CRC_POLY_LOCS[crc_length]
    p = [0] * (crc_length + 1)
    for i in locs:
        p[i] = 1
    return p[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = list(np.asarray(info_bits, dtype=int))
    p = _crc_poly(crc_length)
    work = info_bits + [0] * crc_length
    times = len(info_bits)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[j + i] ^= p[j]
    check_code = work[-crc_length:]
    return np.array(info_bits + check_code, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


def _get_up_loc(bit_matrix, n):
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(len(detect_array)):
        if detect_array[i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to_bit(llr_matrix, bit_matrix, info_set, frozen_bit, target_bit):
    """将 SC 状态推进到 target_bit 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix, n)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, target_bit] not in (0, 1):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]

        if _all_defined(up_bit):
            position = _up(position)
        elif _all_defined(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit
        elif _all_defined(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_set:
                    right_bit = 0 if right_llr[0] > 0 else 1
                else:
                    right_bit = frozen_bit
                bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = right_bit
            else:
                position = _rightdown(position)
        elif _all_defined(left_bit):
            length = len(left_bit)
            right_llr = np.array(
                [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
            )
            llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = right_llr
        elif not _all_defined(left_llr):
            length = span // 2
            left_llr = np.array(
                [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
            )
            llr_matrix[position[0] + 1][position[1]:position[1] + span // 2] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in info_set:
                    left_bit = 0 if left_llr[0] >= 0 else 1
                else:
                    left_bit = frozen_bit
                bit_matrix[position[0] + 1][position[1]:position[1] + span // 2] = left_bit
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _path_metric(llr_vec, bit_vec):
    pm = 0.0
    for llr, bit in zip(llr_vec, bit_vec):
        expected = 0 if llr >= 0 else 1
        if bit != expected:
            pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_pos = _frozen_to_info_pos(frozen_bits)
        self.info_set = set(self.info_pos.tolist())
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bit = 0

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_perm = llr_ch[br]

        n = self.n
        N = self.N
        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_perm

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_positions = [i for i in self.info_pos if not self.frozen_bits[i]]

        for split_pos in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_m, bit_m = _sc_step_to_bit(
                    llr_m.copy(), bit_m.copy(), self.info_set, self.frozen_bit, split_pos
                )
                llr_at = llr_m[n, split_pos] if not np.isnan(llr_m[n, split_pos]) else 0.0

                for bit in (0, 1):
                    bm = bit_m.copy()
                    bm[n, split_pos] = bit
                    penalty = 0.0 if (bit == 0 and llr_at >= 0) or (bit == 1 and llr_at < 0) else abs(llr_at)
                    new_llr.append(llr_m.copy())
                    new_bit.append(bm)
                    new_pm.append(pm + penalty)

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]

        for idx in range(len(llr_list)):
            llr_list[idx], bit_list[idx] = _sc_step_to_bit(
                llr_list[idx], bit_list[idx], self.info_set, self.frozen_bit, N - 1
            )

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for i in order:
                u_cand = np.nan_to_num(bit_list[i][n], nan=0).astype(int)
                if crc_check(u_cand[self.info_pos], self.crc_length):
                    return u_cand, pm_list[i]
                if best_u is None:
                    best_u = u_cand
                    best_pm = pm_list[i]
            return best_u, best_pm

        i = order[0]
        u_hat = np.nan_to_num(bit_list[i][n], nan=0).astype(int)
        return u_hat, pm_list[i]
