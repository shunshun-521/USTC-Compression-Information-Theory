"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from decoder_sc import (
    sc_decode, f_operation, g_operation, _all_computed,
    _leftdown, _rightdown, _up, _get_up_bit,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_update(pm, llr_array, bit_array):
    for llr, bit in zip(llr_array, bit_array):
        expected = 0 if llr >= 0 else 1
        if bit != expected:
            pm += abs(llr)
    return pm


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect = -1
    for i in range(N):
        if bit_matrix[n][i] not in (0, 1):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to_bit(llr_matrix, bit_matrix, frozen_bits, target_bit):
    """运行 SC 直到 bit_matrix[n][target_bit] 被判决。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][target_bit] not in (0, 1):
        layer, col, max_layer, _ = position
        span = 2 ** (max_layer - layer)
        up_llr = llr_matrix[layer][col:col + span]
        up_bit = bit_matrix[layer][col:col + span]
        half = span // 2
        left_llr = llr_matrix[layer + 1][col:col + half]
        left_bit = bit_matrix[layer + 1][col:col + half]
        right_llr = llr_matrix[layer + 1][col + half:col + span]
        right_bit = bit_matrix[layer + 1][col + half:col + span]

        if _all_computed(up_bit):
            position = _up(position)
        else:
            if _all_computed(right_bit):
                bit_matrix[layer][col:col + span] = _get_up_bit(left_bit, right_bit)
            else:
                if _all_computed(right_llr):
                    if layer == max_layer - 1:
                        right_pos = col + 1
                        if frozen_bits[right_pos]:
                            val = 0
                        else:
                            val = 0 if right_llr[0] > 0 else 1
                        bit_matrix[layer + 1][col + half] = val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_computed(left_bit):
                        right_llr_new = g_operation(up_llr[:half], up_llr[half:], left_bit)
                        llr_matrix[layer + 1][col + half:col + span] = right_llr_new
                    else:
                        if not _all_computed(left_llr):
                            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
                            llr_matrix[layer + 1][col:col + half] = left_llr_new
                        else:
                            if layer == max_layer - 1:
                                left_pos = col
                                if frozen_bits[left_pos]:
                                    val = 0
                                else:
                                    val = 0 if left_llr[0] >= 0 else 1
                                bit_matrix[layer + 1][col] = val
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（在信息位处路径分裂）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _init_matrices(self, llr_ch):
        llr_m = np.full((self.n + 1, self.N), np.nan)
        bit_m = np.full((self.n + 1, self.N), np.nan)
        llr_m[0] = llr_ch
        return llr_m, bit_m

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        split_positions = list(self.info_indices)

        llr_list = [self._init_matrices(llr_ch)[0]]
        bit_list = [self._init_matrices(llr_ch)[1]]
        pm_list = [0.0]
        split_loc = 0
        prev_pos = -1

        while split_loc < len(split_positions):
            target = split_positions[split_loc]
            l_now = len(pm_list)
            for i in range(l_now):
                llr_m, bit_m = llr_list[i], bit_list[i]
                pm = pm_list[i]
                llr_m, bit_m = _sc_step_to_bit(llr_m, bit_m, self.frozen_bits, target)
                llr_list[i], bit_list[i] = llr_m, bit_m

                if self.frozen_bits[target]:
                    pm_list[i] = _pm_update(
                        pm,
                        llr_m[n, prev_pos + 1:target + 1],
                        bit_m[n, prev_pos + 1:target + 1],
                    )
                else:
                    llr_seg = llr_m[n, prev_pos + 1:target + 1]
                    bit_seg = bit_m[n, prev_pos + 1:target + 1]
                    pm_list[i] = _pm_update(pm, llr_seg, bit_seg)
                    llr_list.append(llr_m.copy())
                    bit_wrong = bit_m.copy()
                    bit_wrong[n, target] = 1 - int(bit_m[n, target])
                    bit_list.append(bit_wrong)
                    wrong_seg = bit_wrong[n, prev_pos + 1:target + 1]
                    pm_list.append(_pm_update(pm, llr_seg, wrong_seg))

            if len(pm_list) > self.list_size:
                keep = np.argsort(pm_list)[:self.list_size]
                pm_list = [pm_list[i] for i in keep]
                llr_list = [llr_list[i] for i in keep]
                bit_list = [bit_list[i] for i in keep]

            prev_pos = target
            split_loc += 1

        # 完成剩余比特
        if prev_pos < N - 1:
            for i in range(len(pm_list)):
                llr_list[i], bit_list[i] = _sc_step_to_bit(
                    llr_list[i], bit_list[i], self.frozen_bits, N - 1
                )

        if self.crc_length > 0:
            order = np.argsort(pm_list)
            for idx in order:
                u_cand = bit_list[idx][n].astype(int)
                if crc_check(u_cand[self.info_indices], self.crc_length):
                    return u_cand, pm_list[idx]
            best = order[0]
            return bit_list[best][n].astype(int), pm_list[best]

        best = int(np.argmin(pm_list))
        return bit_list[best][n].astype(int), pm_list[best]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    rng = np.random.default_rng(1)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x).astype(float), 1.0) * 1e3
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches} (expect 0)")
    assert mismatches == 0
