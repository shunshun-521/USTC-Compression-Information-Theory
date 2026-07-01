"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _all_num,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_right_llr,
    _get_left_llr,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect = -1
    for i in range(N):
        if np.isnan(bit_matrix[n, i]):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _get_left_bit(left_llr, info_indices, frozen_bits, pos):
    if not frozen_bits[pos]:
        return 0 if left_llr >= 0 else 1
    return 0


def _get_right_bit(right_llr, info_indices, frozen_bits, pos):
    if not frozen_bits[pos]:
        return 0 if right_llr > 0 else 1
    return 0


def _pm_update(llr_array, bit_array):
    pm = 0.0
    for llr, bit in zip(llr_array, bit_array):
        hard = 0 if llr >= 0 else 1
        if hard != int(bit):
            pm += abs(llr)
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, frozen_bits, info_indices, split_pos):
    """SC 译码推进至 split_pos 位完成判决。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]
    max_iter = N * n * 16
    it = 0

    while np.isnan(bit_matrix[n, split_pos]):
        it += 1
        if it > max_iter:
            raise RuntimeError("SCL step timeout")

        layer = position[0]
        idx = position[1]
        max_layer = position[2]
        span = 2 ** (max_layer - layer)

        up_llr = llr_matrix[layer, idx : idx + span]
        up_bit = bit_matrix[layer, idx : idx + span]
        half = span // 2
        left_llr = llr_matrix[layer + 1, idx : idx + half]
        left_bit = bit_matrix[layer + 1, idx : idx + half]
        right_llr = llr_matrix[layer + 1, idx + half : idx + span]
        right_bit = bit_matrix[layer + 1, idx + half : idx + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            bit_matrix[layer, idx : idx + span] = _get_up_bit(left_bit, right_bit)
        elif _all_num(right_llr):
            if layer == max_layer - 1:
                right_pos = idx + half
                bit_matrix[layer + 1, right_pos] = _get_right_bit(
                    right_llr[0], info_indices, frozen_bits, right_pos
                )
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            llr_matrix[layer + 1, idx + half : idx + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_num(left_llr):
            llr_matrix[layer + 1, idx : idx + half] = _get_left_llr(up_llr)
        else:
            if layer == max_layer - 1:
                bit_matrix[layer + 1, idx] = _get_left_bit(
                    left_llr[0], info_indices, frozen_bits, idx
                )
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, L = self.N, self.n, self.list_size

        if L == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_pos = list(self.info_indices)
        split_loc = 0
        prev_end = -1

        while split_loc < len(split_pos):
            phi = split_pos[split_loc]
            new_llr, new_bit, new_pm = [], [], []

            for i in range(len(llr_list)):
                lm = llr_list[i]
                bm = bit_list[i]
                pm = pm_list[i]

                lm2, bm2 = _sc_stepping_decoder(
                    lm.copy(), bm.copy(), self.frozen_bits, self.info_indices, phi
                )
                seg_llr = lm2[n, prev_end + 1 : phi + 1]
                seg_bit = bm2[n, prev_end + 1 : phi + 1]
                pm_add = _pm_update(seg_llr, seg_bit)

                new_llr.append(lm2)
                new_bit.append(bm2)
                new_pm.append(pm + pm_add)

                bm_wrong = bm2.copy()
                bm_wrong[n, phi] = 1 - int(bm2[n, phi])
                new_llr.append(lm2.copy())
                new_bit.append(bm_wrong)
                seg_bit_w = bm_wrong[n, prev_end + 1 : phi + 1]
                new_pm.append(pm + _pm_update(seg_llr, seg_bit_w))

            order = np.argsort(new_pm)[:L]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            split_loc += 1
            prev_end = phi

        if prev_end < N - 1:
            for i in range(len(llr_list)):
                lm2, bm2 = _sc_stepping_decoder(
                    llr_list[i].copy(),
                    bit_list[i].copy(),
                    self.frozen_bits,
                    self.info_indices,
                    N - 1,
                )
                seg_llr = lm2[n, prev_end + 1 : N]
                seg_bit = bm2[n, prev_end + 1 : N]
                llr_list[i] = lm2
                bit_list[i] = bm2
                pm_list[i] += _pm_update(seg_llr, seg_bit)

        order = np.argsort(pm_list)
        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][n].astype(int)
                if crc_check(u_cand[self.info_indices], self.crc_length):
                    return u_cand, pm_list[idx]

        best = order[0]
        return bit_list[best][n].astype(int), pm_list[best]


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=2.0, num_frames=20):
    """L=1 的 SCL 应与 SC 等价。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import (
        awgn_channel,
        bpsk_modulate,
        compute_llr,
        eb_n0_to_sigma,
        prepare_channel_llr,
    )

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = prepare_channel_llr(compute_llr(y, sigma))
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("SCL L=1 != SC")

    return True


if __name__ == "__main__":
    verify_scl_equals_sc(N=32, K=16, num_frames=10)
    print("SCL L=1 verification passed.")
