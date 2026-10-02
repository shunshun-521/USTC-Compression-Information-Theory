"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        reg = 0
        poly = 0x07
        for b in info_bits:
            reg ^= int(b)
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> i) & 1 for i in range(7, -1, -1)], dtype=np.int8)
        return np.concatenate([info_bits, crc_bits])
    if crc_length == 16:
        reg = 0
        poly = 0x8005
        for b in info_bits:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> i) & 1 for i in range(15, -1, -1)], dtype=np.int8)
        return np.concatenate([info_bits, crc_bits])
    raise ValueError("crc_length must be 8 or 16")


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


def _pm_update_hf(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        hard = 0 if llr_array[i] >= 0 else 1
        if int(bit_array[i]) != hard:
            pm += abs(float(llr_array[i]))
    return pm


def _get_up_loc(bit_row):
    detect = -1
    for i, v in enumerate(bit_row):
        if not (v == 0 or v == 1):
            detect = i - 1
            break
    n = int(np.log2(len(bit_row)))
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_stepping(llr_matrix, bit_matrix, info_set, split_pos):
    """SC 译码至 split_pos 判决完成。"""
    from decoder_sc import (
        _all_computed,
        _get_left_bit,
        _get_left_llr,
        _get_right_bit,
        _get_right_llr,
        _get_up_bit,
        _leftdown,
        _rightdown,
        _up,
    )

    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix[n])
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        p0, p1, p2, _ = position
        span = 2 ** (p2 - p0)
        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1][p1 : p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2 : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2 : p1 + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            bit_matrix[p0][p1 : p1 + span] = _get_up_bit(left_bit, right_bit).copy()
        elif _all_computed(right_llr):
            if p0 == p2 - 1:
                rb = _get_right_bit(float(right_llr[0]), info_set, p1 + 1)
                bit_matrix[p0 + 1][p1 + span // 2 : p1 + span] = rb
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            llr_matrix[p0 + 1][p1 + span // 2 : p1 + span] = _get_right_llr(left_bit, up_llr)
        elif not _all_computed(left_llr):
            llr_matrix[p0 + 1][p1 : p1 + span // 2] = _get_left_llr(up_llr)
        elif p0 == p2 - 1:
            lb = _get_left_bit(float(left_llr[0]), info_set, p1)
            bit_matrix[p0 + 1][p1 : p1 + span // 2] = lb
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.sort(np.where(~self.frozen_bits)[0])
        self.info_set = set(int(i) for i in self.info_indices)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        llr0 = np.ones((n + 1, N), dtype=np.float64)
        llr0[llr0 == 1] = np.nan
        bit0 = llr0.copy()
        llr0[0] = llr_ch

        llr_list = [llr0.copy()]
        bit_list = [bit0.copy()]
        pm_list = [0.0]

        split_pos = list(self.info_indices)
        split_loc = 0
        l_now = 1

        while split_loc < len(split_pos):
            sp = split_pos[split_loc]
            prev = split_pos[split_loc - 1] if split_loc > 0 else -1
            new_llr, new_bit, new_pm = [], [], []
            for i in range(l_now):
                lm, bm, pm = llr_list[i], bit_list[i], pm_list[i]
                lm_c, bm_c = _sc_stepping(lm.copy(), bm.copy(), self.info_set, sp)
                sl = bm_c[n][prev + 1 : sp + 1]
                ll = lm_c[n][prev + 1 : sp + 1]
                pm_c = pm + _pm_update_hf(ll, sl)
                llr_list[i], bit_list[i], pm_list[i] = lm_c, bm_c, pm_c

                lm_w, bm_w = lm_c.copy(), bm_c.copy()
                bm_w[n][sp] = 1 - bm_w[n][sp]
                sl_w = bm_w[n][prev + 1 : sp + 1]
                ll_w = lm_c[n][prev + 1 : sp + 1]
                pm_w = pm + _pm_update_hf(ll_w, sl_w)
                new_llr.append(lm_w)
                new_bit.append(bm_w)
                new_pm.append(pm_w)

            llr_list.extend(new_llr)
            bit_list.extend(new_bit)
            pm_list.extend(new_pm)

            if len(pm_list) > self.list_size:
                keep = np.argsort(pm_list)[: self.list_size]
                llr_list = [llr_list[i] for i in keep]
                bit_list = [bit_list[i] for i in keep]
                pm_list = [pm_list[i] for i in keep]

            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for i in range(l_now):
                lm, bm, pm = llr_list[i], bit_list[i], pm_list[i]
                lm, bm = _sc_stepping(lm.copy(), bm.copy(), self.info_set, N - 1)
                prev = split_pos[-1]
                sl = bm[n][prev + 1 : N]
                ll = lm[n][prev + 1 : N]
                pm_list[i] = pm + _pm_update_hf(ll, sl)
                llr_list[i], bit_list[i] = lm, bm

        best_idx = int(np.argmin(pm_list))
        u_hat = bit_list[best_idx][n].astype(int)

        if self.crc_length > 0:
            valid = []
            for idx, bm in enumerate(bit_list):
                ib = bm[n][self.info_indices].astype(int)
                if crc_check(ib, self.crc_length):
                    valid.append(idx)
            if valid:
                best_idx = min(valid, key=lambda i: pm_list[i])
                u_hat = bit_list[best_idx][n].astype(int)

        return u_hat, pm_list[best_idx]
