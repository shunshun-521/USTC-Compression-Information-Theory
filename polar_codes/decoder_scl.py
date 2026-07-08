"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _sc_decode_fsm, precompute_sc_indices


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


def _sc_step_to_next_bit(llr_matrix, bit_matrix, position, frozen_bits, info_set):
    """
  执行 SC FSM 直至下一个比特判决点，返回 (bit_pos, llr_val) 或完成标志。
    """
    N = llr_matrix.shape[1]
    n = int(math.log2(N))

    while True:
        if not np.any(np.isnan(bit_matrix[n])):
            return None, None, llr_matrix, bit_matrix, position

        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1][p1 : p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2 : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2 : p1 + span]

        def all_ok(arr):
            return not np.any(np.isnan(arr))

        if all_ok(up_bit):
            position = _up_pos(position)
        elif all_ok(right_bit):
            merged = np.zeros(span, dtype=int)
            merged[: span // 2] = (left_bit.astype(int) + right_bit.astype(int)) % 2
            merged[span // 2 :] = right_bit.astype(int)
            bit_matrix[p0][p1 : p1 + span] = merged
        elif all_ok(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = position[1] + 1
                return bit_pos, right_llr[0], llr_matrix, bit_matrix, position
            position = _rightdown(position)
        elif all_ok(left_bit.astype(float)):
            from decoder_sc import g_operation

            right_new = np.array(
                [
                    g_operation(up_llr[i], up_llr[i + span // 2], int(left_bit[i]))
                    for i in range(span // 2)
                ]
            )
            llr_matrix[p0 + 1][p1 + span // 2 : p1 + span] = right_new
        elif not all_ok(left_llr):
            from decoder_sc import f_operation

            left_new = np.array(
                [f_operation(up_llr[i], up_llr[i + span // 2]) for i in range(span // 2)]
            )
            llr_matrix[p0 + 1][p1 : p1 + span // 2] = left_new
        else:
            if position[0] == position[2] - 1:
                bit_pos = position[1]
                return bit_pos, left_llr[0], llr_matrix, bit_matrix, position
            position = _leftdown(position)


def _apply_bit_decision(llr_matrix, bit_matrix, position, bit_pos, bit_val):
    """在判决点写入硬判决并继续推进 FSM"""
    n = int(math.log2(llr_matrix.shape[1]))
    span = 1
    p0 = n - 1
    p1 = bit_pos if bit_pos % 2 == 0 else bit_pos - 1
    if bit_pos % 2 == 1:
        bit_matrix[p0 + 1][bit_pos : bit_pos + 1] = bit_val
    else:
        bit_matrix[p0 + 1][bit_pos : bit_pos + 1] = bit_val
    return llr_matrix, bit_matrix, position


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up_pos(pos):
    p0 = pos[0] - 1
    p1 = int(
        np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1)))
        * (2 ** (pos[2] - pos[0] + 1))
    )
    return [p0, p1, pos[2], pos[3]]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.rev]

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = _sc_decode_fsm(llr_br, self.info_indices, frozen_value=0)
            return u_hat, 0.0

        return self._decode_scl(llr_br)

    def _decode_scl(self, llr_br):
        N, n = self.N, self.n
        paths = [
            {
                "pm": 0.0,
                "llr_matrix": np.full((n + 1, N), np.nan, dtype=np.float64),
                "bit_matrix": np.full((n + 1, N), np.nan, dtype=np.float64),
                "position": [0, 0, n, N],
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["llr_matrix"][0] = llr_br.copy()

        while True:
            finished = all(not np.any(np.isnan(p["bit_matrix"][n])) for p in paths)
            if finished:
                break

            new_paths = []
            for path in paths:
                lm = path["llr_matrix"]
                bm = path["bit_matrix"]
                pos = list(path["position"])

                bit_pos, llr_val, lm2, bm2, pos2 = _sc_step_to_next_bit(
                    lm, bm, pos, self.frozen_bits, set(self.info_indices)
                )
                if bit_pos is None:
                    new_paths.append(path)
                    continue

                if self.frozen_bits[bit_pos]:
                    p_copy = copy.deepcopy(path)
                    p_copy["pm"] += _path_metric_penalty(llr_val, 0)
                    p_copy["u_hat"][bit_pos] = 0
                    p_copy["llr_matrix"] = lm2
                    p_copy["bit_matrix"] = bm2
                    p_copy["position"] = pos2
                    n_span = 1
                    p0 = n - 1
                    if bit_pos % 2 == 0:
                        p_copy["bit_matrix"][p0 + 1][bit_pos] = 0
                    else:
                        p_copy["bit_matrix"][p0 + 1][bit_pos] = 0
                    new_paths.append(p_copy)
                else:
                    for u_bit in (0, 1):
                        p_copy = copy.deepcopy(path)
                        p_copy["pm"] += _path_metric_penalty(llr_val, u_bit)
                        p_copy["u_hat"][bit_pos] = u_bit
                        p_copy["llr_matrix"] = copy.deepcopy(lm2)
                        p_copy["bit_matrix"] = copy.deepcopy(bm2)
                        p_copy["position"] = list(pos2)
                        p0 = n - 1
                        p_copy["bit_matrix"][p0 + 1][bit_pos] = u_bit
                        new_paths.append(p_copy)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_paths = []
        if self.crc_length > 0:
            for p in paths:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(p)

        best = min(crc_paths or paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)

    mismatches = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/30")
