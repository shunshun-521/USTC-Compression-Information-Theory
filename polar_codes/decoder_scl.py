"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode_tree, _all_known, _up, _leftdown, _rightdown, _get_up_bit, _get_left_llr, _get_right_llr, _frozen_to_info_pos


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
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
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


def _pm_update(pm, llr, bit):
    hard = 0 if llr > 0 else 1
    if bit != hard:
        pm += abs(llr)
    return pm


def _advance_to_bit(llr_matrix, bit_matrix, phi, info_pos, frozen_val, n, N):
    """将 SC 树状态推进到完成第 phi 个比特判决。"""
    position = [0, 0, n, N]
    info_set = set(info_pos)

    while not (bit_matrix[n, phi] == 0 or bit_matrix[n, phi] == 1):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start : start + span]
        up_bit = bit_matrix[position[0], start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : start + span]
        right_bit = bit_matrix[position[0] + 1, start + half : start + span]

        if _all_known(up_bit):
            position = _up(position)
            continue

        if _all_known(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], start : start + span] = up_bit_val.copy()
            continue

        if _all_known(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = start + half
                if bit_pos in info_set:
                    bit_val = 0 if right_llr[0] > 0 else 1
                else:
                    bit_val = frozen_val
                bit_matrix[position[0] + 1, bit_pos] = bit_val
            else:
                position = _rightdown(position)
            continue

        if _all_known(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half : start + span] = right_llr_val
            continue

        if not _all_known(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start : start + half] = left_llr_val
            continue

        if position[0] == position[2] - 1:
            bit_pos = start
            if bit_pos in info_set:
                bit_val = 0 if left_llr[0] >= 0 else 1
            else:
                bit_val = frozen_val
            bit_matrix[position[0] + 1, bit_pos] = bit_val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix, float(left_llr[0]) if False else 0.0


def _leaf_llr_for_phi(llr_matrix, bit_matrix, phi, info_pos, frozen_val, n, N):
    llr_matrix = llr_matrix.copy()
    bit_matrix = bit_matrix.copy()
    _advance_to_bit(llr_matrix, bit_matrix, phi, info_pos, frozen_val, n, N)
    # 回溯获取该 bit 判决所用 LLR：重新推进并捕获
    position = [0, 0, n, N]
    info_set = set(info_pos)
    current_llr = 0.0
    while not (bit_matrix[n, phi] == 0 or bit_matrix[n, phi] == 1):
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start : start + span]
        up_bit = bit_matrix[position[0], start : start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : start + span]
        right_bit = bit_matrix[position[0] + 1, start + half : start + span]

        if _all_known(up_bit):
            position = _up(position)
            continue
        if _all_known(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], start : start + span] = up_bit_val.copy()
            continue
        if _all_known(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = start + half
                current_llr = float(right_llr[0])
                if bit_pos in info_set:
                    bit_val = 0 if right_llr[0] > 0 else 1
                else:
                    bit_val = frozen_val
                bit_matrix[position[0] + 1, bit_pos] = bit_val
            else:
                position = _rightdown(position)
            continue
        if _all_known(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half : start + span] = right_llr_val
            continue
        if not _all_known(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start : start + half] = left_llr_val
            continue
        if position[0] == position[2] - 1:
            bit_pos = start
            current_llr = float(left_llr[0])
            if bit_pos in info_set:
                bit_val = 0 if left_llr[0] >= 0 else 1
            else:
                bit_val = frozen_val
            bit_matrix[position[0] + 1, bit_pos] = bit_val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix, current_llr


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.info_pos = self.info_indices.tolist()

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_tree(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        n, N = self.n, self.N
        llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr0[0] = llr_ch

        paths = [(llr0.copy(), bit0.copy(), 0.0)]

        for phi in range(N):
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m2, bit_m2, llr_val = _leaf_llr_for_phi(
                    llr_m, bit_m, phi, self.info_pos, 0, n, N
                )
                bit_val = int(bit_m2[n, phi])

                if self.frozen_bits[phi]:
                    new_paths.append((llr_m2, bit_m2, _pm_update(pm, llr_val, 0)))
                else:
                    for candidate in (0, 1):
                        lm = llr_m2.copy()
                        bm = bit_m2.copy()
                        bm[n, phi] = candidate
                        new_paths.append((lm, bm, _pm_update(pm, llr_val, candidate)))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        candidates = []
        for llr_m, bit_m, pm in paths:
            u_hat = bit_m[n].astype(int)
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            pm, u_hat = min(valid if valid else candidates, key=lambda x: x[0])
        else:
            pm, u_hat = min(candidates, key=lambda x: x[0])

        return u_hat.copy(), pm
