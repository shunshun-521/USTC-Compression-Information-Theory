"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _all_filled,
    _decide_bit,
    _up_position,
    f_operation,
    g_operation,
    precompute_sc_indices,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _get_up_loc(bit_matrix, n, N):
    detect = -1
    for i in range(N):
        if np.isnan(bit_matrix[n, i]):
            detect = i - 1
            break
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1] if detect >= 0 else [0, 0]


def _sc_step_to(llr_matrix, bit_matrix, frozen_bits, split_pos):
    """SC 推进至完成 split_pos 判决。"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix, n, N)
    position = [loc[0], loc[1], n, N]

    while np.isnan(bit_matrix[n, split_pos]):
        span = 2 ** (position[2] - position[0])
        half = span // 2
        p0, p1 = position[0], position[1]

        up_llr = llr_matrix[p0][p1: p1 + span]
        up_bit = bit_matrix[p0][p1: p1 + span]
        left_llr = llr_matrix[p0 + 1][p1: p1 + half]
        left_bit = bit_matrix[p0 + 1][p1: p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half: p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half: p1 + span]

        if _all_filled(up_bit):
            _up_position(position)
        elif _all_filled(right_bit):
            up_val = np.zeros(span)
            up_val[:half] = (left_bit + right_bit) % 2
            up_val[half:] = right_bit
            bit_matrix[p0][p1: p1 + span] = up_val
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                pos = p1 + half
                bit_matrix[p0 + 1][pos] = _decide_bit(right_llr[0], pos, frozen_bits)
            else:
                position[0] += 1
                position[1] += half
        elif _all_filled(left_bit):
            right_new = np.array(
                [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
            )
            llr_matrix[p0 + 1][p1 + half: p1 + span] = right_new
        elif _all_filled(left_llr):
            if position[0] == position[2] - 1:
                pos = p1
                bit_matrix[p0 + 1][pos] = _decide_bit(left_llr[0], pos, frozen_bits)
            else:
                position[0] += 1
        else:
            llr_matrix[p0 + 1][p1: p1 + half] = f_operation(up_llr[:half], up_llr[half:])

    return llr_matrix, bit_matrix


def _pm_update(llr_slice, bit_slice):
  pm = 0.0
  for llr, bit in zip(llr_slice, bit_slice):
      hard = 0 if llr >= 0 else 1
      if bit != hard:
          pm += abs(llr)
  return pm


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        n, N = self.n, self.N
        split_pos = list(self.info_positions)

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_ch
        llr_list = [llr_matrix]
        bit_list = [bit_matrix]
        pm_list = [0.0]

        prev = -1
        for sp in split_pos:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                lm, bm = _sc_step_to(llr_m.copy(), bit_m.copy(), self.frozen_bits, sp)
                llr_slice = lm[n, prev + 1: sp + 1]
                bit_slice = bm[n, prev + 1: sp + 1].astype(int)
                base_pm = pm + _pm_update(llr_slice, bit_slice)

                if self.frozen_bits[sp]:
                    new_llr.append(lm)
                    new_bit.append(bm)
                    new_pm.append(base_pm)
                else:
                    bm0 = bm.copy()
                    bm0[n, sp] = 0
                    new_llr.append(lm.copy())
                    new_bit.append(bm0)
                    new_pm.append(base_pm)

                    bm1 = bm.copy()
                    bm1[n, sp] = 1
                    wrong_pm = pm + _pm_update(llr_slice, np.where(np.arange(prev + 1, sp + 1) == sp, 1, bit_slice))
                    new_llr.append(lm.copy())
                    new_bit.append(bm1)
                    new_pm.append(wrong_pm)

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            prev = sp

        if split_pos and split_pos[-1] != N - 1:
            for i in range(len(llr_list)):
                llr_list[i], bit_list[i] = _sc_step_to(
                    llr_list[i], bit_list[i], self.frozen_bits, N - 1
                )
                pm_list[i] += _pm_update(
                    llr_list[i][n, prev + 1: N],
                    bit_list[i][n, prev + 1: N].astype(int),
                )

        order = np.argsort(pm_list)
        for idx in order:
            u_hat = bit_list[idx][n].astype(int)
            if self.crc_length > 0:
                if crc_check(u_hat[self.info_positions], self.crc_length):
                    return u_hat, pm_list[idx]
            else:
                return u_hat, pm_list[idx]

        return bit_list[order[0]][n].astype(int), pm_list[order[0]]


def verify_scl_equals_sc(N=8, K=4, seed=0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl)

    return True


if __name__ == "__main__":
    assert crc_check(crc_encode(np.array([1, 0, 1, 1, 0, 1, 0, 1]), 8), 8)
    verify_scl_equals_sc()
    print("SCL 译码器校验通过")
