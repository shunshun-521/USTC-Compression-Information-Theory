"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_decided,
    _frozen_to_info_pos,
    _get_left_bit,
    _get_left_llr,
    _get_right_bit,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    f_operation,
    g_operation,
)

def _crc_poly(crc_length):
    """生成 CRC 多项式系数（GF(2) 长除法）"""
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError(f"不支持的 CRC 长度: {crc_length}")
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def _crc_division(info_bits, crc_length):
    """CRC 长除法，返回校验位"""
    p = _crc_poly(crc_length)
    info = [int(b) for b in info_bits]
    times = len(info)
    for _ in range(crc_length):
        info.append(0)
    for i in range(times):
        if info[i] == 1:
            for j in range(crc_length + 1):
                info[j + i] ^= p[j]
    return info[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    check = _crc_division(info_bits, crc_length)
    return np.concatenate([info_bits, np.array(check, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = [int(b) for b in bits]
    info_len = len(bits) - crc_length
    expected = list(crc_encode(np.array(bits[:info_len], dtype=int), crc_length))
    return all(bits[i] == expected[i] for i in range(len(bits)))


def _pm_update(llr, bits):
    """路径度量更新（min-sum 风格）"""
    pm = 0.0
    for l, b in zip(llr, bits):
        hard = 0 if l >= 0 else 1
        if hard != b:
            pm += abs(l)
    return pm


def _sc_step_to_split(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 单步推进至完成 split_pos 处判决"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    loc_row = n - 1
    loc_col = split_pos
    if not np.isnan(bit_matrix[n, split_pos]):
        return llr_matrix, bit_matrix

    detect = split_pos - 1
    for i in range(N):
        if np.isnan(bit_matrix[n, i]):
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row, loc_col = n - 1, max(detect, 0)
    else:
        loc_row, loc_col = n - 1, max(detect - 1, 0)
    if detect == -1:
        loc_row, loc_col = 0, 0

    position = [loc_row, loc_col, n, N]

    while np.isnan(bit_matrix[n, split_pos]):
        block = 2 ** (position[2] - position[0])
        c, r = position[1], position[0]
        up_llr = llr_matrix[r, c:c + block]
        up_bit = bit_matrix[r, c:c + block]
        left_llr = llr_matrix[r + 1, c:c + block // 2]
        left_bit = bit_matrix[r + 1, c:c + block // 2]
        right_llr = llr_matrix[r + 1, c + block // 2:c + block]
        right_bit = bit_matrix[r + 1, c + block // 2:c + block]

        if _all_decided(up_bit):
            position = _up(position)
        elif _all_decided(right_bit):
            bit_matrix[r, c:c + block] = _get_up_bit(left_bit, right_bit)
        elif _all_decided(right_llr):
            if r == position[2] - 1:
                idx = c + block // 2
                bit_matrix[r + 1, idx] = _get_right_bit(
                    right_llr, information_pos, frozen_bit, idx
                )
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            llr_matrix[r + 1, c + block // 2:c + block] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_decided(left_llr):
            llr_matrix[r + 1, c:c + block // 2] = _get_left_llr(up_llr)
        else:
            if r == position[2] - 1:
                bit_matrix[r + 1, c] = _get_left_bit(
                    left_llr, information_pos, frozen_bit, c
                )
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = _frozen_to_info_pos(self.frozen_bits)
        self.frozen_bit = 0

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        info_pos = self.information_pos

        llr_list = [np.full((n + 1, N), np.nan)]
        llr_list[0][0] = llr_ch
        bit_list = [np.full((n + 1, N), np.nan)]
        pm_list = [0.0]

        split_positions = [i for i in info_pos]
        if split_positions and split_positions[-1] != N - 1:
            split_positions.append(N - 1)

        for split_pos in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            prev_start = (
                split_positions[split_positions.index(split_pos) - 1]
                if split_pos in split_positions and split_positions.index(split_pos) > 0
                else -1
            )

            for idx in range(len(llr_list)):
                lm = llr_list[idx].copy()
                bm = bit_list[idx].copy()
                pm = pm_list[idx]

                lm, bm = _sc_step_to_split(
                    lm, bm, info_pos, self.frozen_bit, split_pos
                )

                if split_pos in info_pos:
                    llr_val = lm[n, split_pos]
                    right_bit = 0 if llr_val >= 0 else 1
                    wrong_bit = 1 - right_bit

                    bm_wrong = bm.copy()
                    bm_wrong[n, split_pos] = wrong_bit

                    seg_start = prev_start + 1
                    seg_llr = lm[n, seg_start:split_pos + 1]
                    seg_bits = bm[n, seg_start:split_pos + 1]
                    seg_wrong = bm_wrong[n, seg_start:split_pos + 1]

                    new_llr.append(lm)
                    new_bit.append(bm)
                    new_pm.append(pm + _pm_update(seg_llr, seg_bits))

                    new_llr.append(lm.copy())
                    new_bit.append(bm_wrong)
                    new_pm.append(pm + _pm_update(seg_llr, seg_wrong))
                else:
                    seg_start = prev_start + 1
                    seg_llr = lm[n, seg_start:split_pos + 1]
                    seg_bits = bm[n, seg_start:split_pos + 1]
                    new_llr.append(lm)
                    new_bit.append(bm)
                    new_pm.append(pm + _pm_update(seg_llr, seg_bits))

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]

        best_idx = int(np.argmin(pm_list))
        u_hat = bit_list[best_idx][n].astype(int)

        if self.crc_length > 0:
            valid = []
            for i, bm in enumerate(bit_list):
                payload = bm[n, info_pos].astype(int)
                if crc_check(payload, self.crc_length):
                    valid.append((pm_list[i], i))
            if valid:
                best_idx = min(valid, key=lambda x: x[0])[1]
                u_hat = bit_list[best_idx][n].astype(int)

        return u_hat, pm_list[best_idx]
