"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    sc_decode,
    f_operation,
    g_operation,
    _all_filled,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    _init_matrices,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if int(bit) != hard:
            pm += abs(llr)
    return pm


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    if np.all(np.isnan(detect_array)):
        return 0, 0
    detect = 0
    for i in range(N):
        if np.isnan(detect_array[i]):
            detect = i - 1
            break
    else:
        detect = N - 1
    if detect < 0:
        return 0, 0
    if detect % 2 == 0:
        return n - 1, detect
    return n - 1, detect - 1


def _sc_step_to(llr_matrix, bit_matrix, frozen_bits, split_pos):
    """推进 SC 状态至 split_pos 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc_row, loc_col = _get_up_loc(bit_matrix)
    position = [loc_row, loc_col, n, N]

    while np.isnan(bit_matrix[n, split_pos]):
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        span = 2 ** (position[2] - position[0])
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[position[0]][position[1] : position[1] + span] = up
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = position[1] + half
                val = 0 if frozen_bits[bit_pos] else (0 if right_llr[0] >= 0 else 1)
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = np.array(
                [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
            )
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = np.array(
                [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)]
            )
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                bit_pos = position[1]
                val = 0 if frozen_bits[bit_pos] else (0 if left_llr[0] >= 0 else 1)
                bit_matrix[position[0] + 1][position[1] : position[1] + half] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        llr0, bit0 = _init_matrices(llr_ch, n, N)
        llr_list = [llr0.copy()]
        bit_list = [bit0.copy()]
        pm_list = [0.0]

        split_positions = list(self.info_positions)
        prev = -1
        for split_pos in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            for idx in range(len(llr_list)):
                llr_m = llr_list[idx].copy()
                bit_m = bit_list[idx].copy()
                pm0 = pm_list[idx]
                llr_m, bit_m = _sc_step_to(llr_m, bit_m, self.frozen_bits, split_pos)
                llr_slice = llr_m[n, prev + 1 : split_pos + 1]
                bit_slice = bit_m[n, prev + 1 : split_pos + 1].astype(int)

                for u_val in (int(bit_m[n, split_pos]), 1 - int(bit_m[n, split_pos])):
                    bm = bit_m.copy()
                    bm[n, split_pos] = u_val
                    new_llr.append(llr_m.copy())
                    new_bit.append(bm)
                    trial_bits = bm[n, prev + 1 : split_pos + 1].astype(int)
                    trial_bits[-1] = u_val
                    new_pm.append(pm0 + _pm_update(llr_slice, trial_bits))

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            prev = split_pos

        if split_positions and split_positions[-1] != N - 1:
            for idx in range(len(llr_list)):
                llr_list[idx], bit_list[idx] = _sc_step_to(
                    llr_list[idx], bit_list[idx], self.frozen_bits, N - 1
                )
                pm_list[idx] += _pm_update(
                    llr_list[idx][n, prev + 1 : N],
                    bit_list[idx][n, prev + 1 : N].astype(int),
                )

        best_idx = int(np.argmin(pm_list))
        u_hat = bit_list[best_idx][n].astype(int)

        if self.crc_length > 0:
            order = np.argsort(pm_list)
            for idx in order:
                u_try = bit_list[idx][n].astype(int)
                if crc_check(u_try, self.crc_length):
                    u_hat = u_try
                    best_idx = idx
                    break

        return u_hat, pm_list[best_idx]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(8.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"L=1 SCL vs SC mismatches: {mismatches}/20")
