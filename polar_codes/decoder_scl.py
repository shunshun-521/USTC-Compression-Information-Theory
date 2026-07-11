"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _all_num,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_right_bit,
    _get_left_bit,
    _get_right_llr,
    _get_left_llr,
)


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] == 1 or detect_array[i] == 0:
            pass
        else:
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row = n - 1
        loc_col = detect
    else:
        loc_row = n - 1
        loc_col = detect - 1
    if detect == -1:
        loc_row = 0
        loc_col = 0
    return [loc_row, loc_col]


def _get_pm_update(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        hard = 0 if llr_array[i] >= 0 else 1
        if hard != bit_array[i]:
            pm += abs(llr_array[i])
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix.shape[1])
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start : start + span]
        up_bit = bit_matrix[position[0], start : start + span]
        half = 2 ** (position[2] - position[0] - 1)
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : start + span]
        right_bit = bit_matrix[position[0] + 1, start + half : start + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], start : start + span] = up_bit
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1, start + half : start + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half : start + span] = right_llr_new
        elif _all_num(left_llr) == 0:
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start : start + half] = left_llr_new
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            left_bit_val = _get_left_bit(
                left_llr[0], information_pos, frozen_bit, left_bit_pos
            )
            bit_matrix[position[0] + 1, start : start + half] = left_bit_val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError("CRC length must be 8 or 16")
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = [int(b) for b in info_bits]
    p = _crc_poly_bits(crc_length)
    work = info_bits.copy()
    times = len(work)
    for _ in range(crc_length):
        work.append(0)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[j + i] ^= p[j]
    check = work[-crc_length:]
    return np.array(info_bits + check, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    bits = [int(b) for b in bits]
    info_len = len(bits) - crc_length
    if info_len <= 0:
        return False
    recoded = crc_encode(bits[:info_len], crc_length)
    return recoded.tolist() == bits


def scl_decode(llr_ch, frozen_bits, list_size=4, crc_length=0):
    """SCL 译码核心函数。"""
    if list_size == 1 and crc_length == 0:
        u_hat = sc_decode(llr_ch, frozen_bits)
        return u_hat, 0.0

    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(y_llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = list(np.where(~frozen_bits)[0])
    frozen_bit = 0

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr

    llr_list = [llr_matrix.copy()]
    bit_list = [bit_matrix.copy()]
    pm_list = [0.0]
    split_pos = information_pos
    split_loc = 0
    l_now = 1

    while split_loc < len(split_pos):
        prev = -1 if split_loc == 0 else split_pos[split_loc - 1]
        cur = split_pos[split_loc]
        new_llr_list = []
        new_bit_list = []
        new_pm_list = []

        for i in range(l_now):
            llr_temp = llr_list[i].copy()
            bit_temp = bit_list[i].copy()
            pm_temp = pm_list[i]
            llr_temp, bit_temp = _sc_stepping_decoder(
                llr_temp, bit_temp, information_pos, frozen_bit, cur
            )
            llr_slice = llr_temp[n, prev + 1 : cur + 1]
            bit_slice = bit_temp[n, prev + 1 : cur + 1]
            pm_update = _get_pm_update(llr_slice, bit_slice)

            new_llr_list.append(llr_temp)
            new_bit_list.append(bit_temp)
            new_pm_list.append(pm_temp + pm_update)

            bit_wrong = bit_temp.copy()
            bit_wrong[n, cur] = 1 - bit_wrong[n, cur]
            wrong_slice = bit_wrong[n, prev + 1 : cur + 1]
            pm_wrong = _get_pm_update(llr_slice, wrong_slice)
            new_llr_list.append(llr_temp.copy())
            new_bit_list.append(bit_wrong)
            new_pm_list.append(pm_temp + pm_wrong)

        order = np.argsort(new_pm_list)
        keep = order[:list_size]
        llr_list = [new_llr_list[i] for i in keep]
        bit_list = [new_bit_list[i] for i in keep]
        pm_list = [new_pm_list[i] for i in keep]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos and split_pos[-1] != N - 1:
        prev = split_pos[-1]
        for i in range(l_now):
            llr_temp = llr_list[i].copy()
            bit_temp = bit_list[i].copy()
            pm_temp = pm_list[i]
            llr_temp, bit_temp = _sc_stepping_decoder(
                llr_temp, bit_temp, information_pos, frozen_bit, N - 1
            )
            llr_slice = llr_temp[n, prev + 1 : N]
            bit_slice = bit_temp[n, prev + 1 : N]
            pm_list[i] = pm_temp + _get_pm_update(llr_slice, bit_slice)
            llr_list[i] = llr_temp
            bit_list[i] = bit_temp

    order = np.argsort(pm_list)
    best_u = None
    best_pm = pm_list[order[0]]

    if crc_length > 0:
        for idx in order:
            u_cand = bit_list[idx][n].astype(int)
            info_bits = u_cand[information_pos]
            if crc_check(info_bits, crc_length):
                best_u = u_cand
                best_pm = pm_list[idx]
                break
        if best_u is None:
            best_u = bit_list[order[0]][n].astype(int)
    else:
        best_u = bit_list[order[0]][n].astype(int)

    return best_u, best_pm


class SCLDecoder:
    """SCL 译码器封装。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        return scl_decode(
            llr_ch,
            self.frozen_bits,
            list_size=self.list_size,
            crc_length=self.crc_length,
        )


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(1)

    mismatches = 0
    for _ in range(30):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_full)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"SCL L=1 vs SC mismatch: {mismatches}/30"
    print("SCL decoder tests passed.")
