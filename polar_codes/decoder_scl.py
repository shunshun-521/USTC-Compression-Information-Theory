"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import sc_decode


_CRC_POLY = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    if crc_length not in _CRC_POLY:
        raise ValueError("crc_length must be 8 or 16")
    poly = _CRC_POLY[crc_length]
    reg = 0
    info_bits = np.asarray(info_bits, dtype=int)
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    encoded = crc_encode(np.asarray(bits[:-crc_length], dtype=int), crc_length)
    return np.array_equal(encoded, np.asarray(bits, dtype=int))


def _path_metric(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _sc_step_until_phi(llr_matrix, bit_matrix, frozen_bits, phi_target):
    """推进 SC 状态直至 bit_matrix[n, phi_target] 已判决。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    information_pos = np.where(~frozen_bits)[0].tolist()
    frozen_bit = 0
    position = [0, 0, n, N]

    def f_hf(L1, L2):
        s1 = np.sign(L1)
        s2 = np.sign(L2)
        s1 = 1 if s1 == 0 else s1
        s2 = 1 if s2 == 0 else s2
        return s1 * s2 * np.min([np.abs(L1), np.abs(L2)])

    def g_op(L1, L2, u1):
        return (1 - 2 * u1) * L1 + L2

    def all_num(x):
        return not np.any(np.isnan(x))

    def leftdown(p):
        return [p[0] + 1, p[1], p[2], p[3]]

    def rightdown(p):
        return [p[0] + 1, p[1] + 2 ** (p[2] - 1 - p[0]), p[2], p[3]]

    def up_move(p):
        p1 = int(np.floor(p[1] / (2 ** (p[2] - p[0] + 1))) * (2 ** (p[2] - p[0] + 1)))
        return [p[0] - 1, p1, p[2], p[3]]

    def get_up_bit(left_b, right_b):
        length = len(left_b)
        temp = np.array([(left_b + right_b) % 2, right_b])
        temp.resize((1, 2 * length))
        return temp[0]

    def get_right_bit(llr_val, pos):
        if frozen_bits[pos]:
            return frozen_bit
        if pos in information_pos:
            return 0 if llr_val > 0 else 1
        return frozen_bit

    def get_left_bit(llr_val, pos):
        if frozen_bits[pos]:
            return frozen_bit
        if pos in information_pos:
            return 0 if llr_val >= 0 else 1
        return frozen_bit

    def get_right_llr(left_b, up):
        half = len(up) // 2
        return np.array([g_op(up[i], up[i + half], left_b[i]) for i in range(half)])

    def get_left_llr(up):
        half = len(up) // 2
        return np.array([f_hf(up[i], up[i + half]) for i in range(half)])

    while not (bit_matrix[n, phi_target] == 0 or bit_matrix[n, phi_target] == 1):
        span = 2 ** (position[2] - position[0])
        col = position[1]
        up_llr = llr_matrix[position[0], col:col + span]
        up_bit = bit_matrix[position[0], col:col + span]
        left_llr = llr_matrix[position[0] + 1, col:col + span // 2]
        left_bit = bit_matrix[position[0] + 1, col:col + span // 2]
        right_llr = llr_matrix[position[0] + 1, col + span // 2:col + span]
        right_bit = bit_matrix[position[0] + 1, col + span // 2:col + span]

        if all_num(up_bit):
            position = up_move(position)
        elif all_num(right_bit):
            bit_matrix[position[0], col:col + span] = get_up_bit(left_bit, right_bit)
        elif all_num(right_llr):
            if position[0] == position[2] - 1:
                idx = col + span // 2
                bit_matrix[position[0] + 1, idx] = get_right_bit(right_llr[0], idx)
            else:
                position = rightdown(position)
        elif all_num(left_bit):
            llr_matrix[position[0] + 1, col + span // 2:col + span] = get_right_llr(
                left_bit, up_llr
            )
        elif all_num(left_llr) == 0:
            llr_matrix[position[0] + 1, col:col + span // 2] = get_left_llr(up_llr)
        elif position[0] == position[2] - 1:
            idx = col
            bit_matrix[position[0] + 1, idx] = get_left_bit(left_llr[0], idx)
        else:
            position = leftdown(position)

    return llr_matrix, bit_matrix


def _init_matrices(llr_ch, N, n):
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr_ch
    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制矩阵）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [{
            "llr": _init_matrices(llr_ch, self.N, self.n)[0],
            "bit": _init_matrices(llr_ch, self.N, self.n)[1],
            "pm": 0.0,
        }]

        for phi in range(self.N):
            expanded = []
            for path in paths:
                llr_m, bit_m = _sc_step_until_phi(
                    path["llr"].copy(), path["bit"].copy(), self.frozen_bits, phi
                )
                cur_llr = llr_m[0, 0] if phi == 0 else llr_m[self.n, phi]
                decided = int(bit_m[self.n, phi])

                if self.frozen_bits[phi]:
                    expanded.append({
                        "llr": llr_m,
                        "bit": bit_m,
                        "pm": path["pm"] + _path_metric(cur_llr, 0),
                    })
                else:
                    for bit in (0, 1):
                        bm = bit_m.copy()
                        bm[self.n, phi] = bit
                        expanded.append({
                            "llr": llr_m.copy(),
                            "bit": bm,
                            "pm": path["pm"] + _path_metric(cur_llr, bit),
                        })

            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]

        paths.sort(key=lambda p: p["pm"])

        if self.crc_length > 0:
            for path in paths:
                u_hat = path["bit"][self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, path["pm"]

        best = paths[0]
        return best["bit"][self.n].astype(int), best["pm"]
