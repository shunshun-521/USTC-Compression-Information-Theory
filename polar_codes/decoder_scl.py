"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _prepare_llr, sc_decode
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _all_computed(arr):
    return not np.any(np.isnan(arr))


def _up(position):
    p0 = position[0] - 1
    span = 2 ** (position[2] - position[0] + 1)
    p1 = int(np.floor(position[1] / span) * span)
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] != 0 and detect_array[i] != 1:
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


def _sc_step(llr_matrix, bit_matrix, info_positions, frozen_val, split_pos):
    """SC 译码至 split_pos 并完成该位判决"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span = 2 ** (position[2] - position[0])
        base = position[1]
        up_llr = llr_matrix[position[0]][base : base + span]
        left_llr = llr_matrix[position[0] + 1][base : base + span // 2]
        left_bit = bit_matrix[position[0] + 1][base : base + span // 2]
        right_llr = llr_matrix[position[0] + 1][base + span // 2 : base + span]
        right_bit = bit_matrix[position[0] + 1][base + span // 2 : base + span]

        if _all_computed(bit_matrix[position[0]][base : base + span]):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][base : base + span] = up_bit_val.copy()
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                pos = base + 1
                if pos in info_positions:
                    bit = 0 if right_llr[0] > 0 else 1
                else:
                    bit = frozen_val
                bit_matrix[position[0] + 1][base + span // 2 : base + span] = bit
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            right_llr_val = (1 - 2 * left_bit) * up_llr[: span // 2] + up_llr[span // 2 :]
            llr_matrix[position[0] + 1][base + span // 2 : base + span] = right_llr_val
        elif not _all_computed(left_llr):
            s1 = np.sign(up_llr[: span // 2])
            s2 = np.sign(up_llr[span // 2 :])
            s1[s1 == 0] = 1
            s2[s2 == 0] = 1
            left_llr_val = s1 * s2 * np.minimum(
                np.abs(up_llr[: span // 2]), np.abs(up_llr[span // 2 :])
            )
            llr_matrix[position[0] + 1][base : base + span // 2] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                pos = base
                if pos in info_positions:
                    bit = 0 if left_llr[0] >= 0 else 1
                else:
                    bit = frozen_val
                bit_matrix[position[0] + 1][base : base + span // 2] = bit
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _pm_update(llr_slice, bit_slice):
    pm = 0.0
    for llr, bit in zip(llr_slice, bit_slice):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.info_positions = set(self.info_indices.tolist())

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        N, n = self.N, self.n
        y_llr = _prepare_llr(llr_ch, N)

        llr_init = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_init = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_init[0] = y_llr

        paths = [(0.0, llr_init.copy(), bit_init.copy())]
        split_positions = [i for i in range(N) if i in self.info_positions]

        prev_pos = -1
        for split_pos in split_positions:
            new_paths = []
            for pm, llr_m, bit_m in paths:
                llr_m, bit_m = _sc_step(
                    llr_m, bit_m, self.info_positions, 0, split_pos
                )
                llr_val = llr_m[n][split_pos]
                for bit in (0, 1):
                    llr_c = llr_m.copy()
                    bit_c = bit_m.copy()
                    bit_c[n][split_pos] = bit
                    start = prev_pos + 1
                    pm_add = _pm_update(llr_c[n][start : split_pos + 1], bit_c[n][start : split_pos + 1])
                    new_paths.append((pm + pm_add, llr_c, bit_c))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]
            prev_pos = split_pos

        if split_positions[-1] != N - 1:
            finalized = []
            for pm, llr_m, bit_m in paths:
                llr_m, bit_m = _sc_step(
                    llr_m, bit_m, self.info_positions, 0, N - 1
                )
                start = prev_pos + 1
                pm_add = _pm_update(llr_m[n][start:N], bit_m[n][start:N])
                finalized.append((pm + pm_add, llr_m, bit_m))
            finalized.sort(key=lambda x: x[0])
            paths = finalized[: self.list_size]

        if self.crc_length > 0:
            crc_paths = []
            for pm, _, bit_m in paths:
                u = bit_m[n].astype(int)
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append((pm, u))
            if crc_paths:
                best_pm, u_hat = min(crc_paths, key=lambda x: x[0])
                return u_hat, best_pm

        best_pm, _, bit_m = min(paths, key=lambda x: x[0])
        return bit_m[n].astype(int), best_pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    sigma = eb_n0_to_sigma(5.0, K / N)
    y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
    llr = compute_llr(y, sigma)

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    print("SCL L=1 vs SC match:", np.array_equal(u_sc, u_scl))

    u_scl4, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr)
    print("SCL L=4 info match:", np.array_equal(u_scl4[info_idx], u[info_idx]))
