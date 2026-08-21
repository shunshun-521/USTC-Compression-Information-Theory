"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import align_llr_for_decoder, _sc_tree_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _path_metric(llr_val, bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if bit == hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器：在信息位逐位扩展路径"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0].astype(int)

    def decode(self, llr_ch):
        llr = align_llr_for_decoder(np.asarray(llr_ch, dtype=np.float64))
        N, n = self.N, self.n
        info_pos = self.info_indices.tolist()

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr

        paths = [(llr_matrix.copy(), bit_matrix.copy(), 0.0)]
        info_list = info_pos if self.crc_length == 0 else info_pos

        for info_idx in info_list:
            new_paths = []
            for llr_m, bit_m, pm in paths:
                partial = self._decode_to_bit(llr_m, bit_m.copy(), info_idx, info_pos)
                llr_p, bit_p, leaf_llr = partial
                for u in (0, 1):
                    bm = bit_p.copy()
                    bm[n, info_idx] = u
                    penalty = _path_metric(leaf_llr, u)
                    new_paths.append((llr_p.copy(), bm, pm + penalty))
            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda x: x[2])
        if self.crc_length > 0:
            valid = []
            for _, bit_m, pm in paths:
                u_hat = bit_m[n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((bit_m, pm))
            if valid:
                u_hat = valid[0][0][n].astype(int)
                return u_hat, valid[0][1]

        u_hat = paths[0][1][n].astype(int)
        return u_hat, paths[0][2]

    def _decode_to_bit(self, llr_matrix, bit_matrix, target_bit, information_pos):
        from decoder_sc import (
            _all_num,
            _get_left_bit,
            _get_left_llr,
            _get_right_bit,
            _get_right_llr,
            _get_up_bit,
            _leftdown,
            _rightdown,
            _up,
        )

        N, n = self.N, self.n
        info = set(information_pos)
        position = [0, 0, n, N]

        while not _all_num(bit_matrix[n, :target_bit + 1]):
            up_llr = llr_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ]
            up_bit = bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ]
            left_llr = llr_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ]
            left_bit = bit_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ]
            right_llr = llr_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ]
            right_bit = bit_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ]

            if _all_num(up_bit):
                position = _up(position)
            elif _all_num(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit_new.copy()
            elif _all_num(right_llr):
                if position[0] == position[2] - 1:
                    pos = position[1] + 1
                    rb = _get_right_bit(right_llr[0], info, 0, pos)
                    bit_matrix[position[0] + 1][
                        position[1]
                        + 2 ** (position[2] - position[0] - 1) : position[1]
                        + 2 ** (position[2] - position[0])
                    ] = rb
                else:
                    position = _rightdown(position)
            elif _all_num(left_bit):
                rr = _get_right_llr(left_bit, up_llr)
                llr_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = rr
            else:
                if not _all_num(left_llr):
                    llr_matrix[position[0] + 1][
                        position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                    ] = _get_left_llr(up_llr)
                else:
                    if position[0] == position[2] - 1:
                        pos = position[1]
                        lb = _get_left_bit(left_llr[0], info, 0, pos)
                        bit_matrix[position[0] + 1][
                            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                        ] = lb
                    else:
                        position = _leftdown(position)

        leaf_llr = llr_matrix[n, target_bit]
        if np.isnan(leaf_llr):
            leaf_llr = 0.0
        return llr_matrix, bit_matrix, float(leaf_llr)
