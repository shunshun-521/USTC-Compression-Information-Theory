"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import _prepare_llr, _tree_sc_decode, f_operation, g_operation, sc_decode


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length not in CRC_POLYNOMIALS:
        raise ValueError("crc_length must be 8 or 16")

    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)

    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) & mask) ^ poly
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _pm_penalty(llr, bit):
    """路径度量惩罚：比特与 LLR 符号不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class _Path:
    __slots__ = ("llr_matrix", "bit_matrix", "pm", "u_hat")

    def __init__(self, n, N, llr):
        self.llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        self.bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        self.llr_matrix[0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_positions = np.where(self.frozen_bits == 0)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = _prepare_llr(llr_ch)
        paths = [_Path(self.n, self.N, llr.copy())]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                u_partial, leaf_llr, llr_m, bit_m = self._decode_to_phase(path, phi)
                if self.frozen_bits[phi]:
                    path.pm += _pm_penalty(leaf_llr, 0)
                    path.u_hat = u_partial
                    path.u_hat[phi] = 0
                    path.llr_matrix = llr_m
                    path.bit_matrix = bit_m
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = copy.copy(path)
                        p.llr_matrix = llr_m.copy()
                        p.bit_matrix = bit_m.copy()
                        p.u_hat = u_partial.copy()
                        p.u_hat[phi] = bit
                        p.pm += _pm_penalty(leaf_llr, bit)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat, path.pm
        return paths[0].u_hat, paths[0].pm

    def _decode_to_phase(self, path, phi):
        from decoder_sc import _all_num, _get_up_bit, _leftdown, _rightdown, _up

        llr_matrix = path.llr_matrix
        bit_matrix = path.bit_matrix
        n = self.n
        N = self.N
        information_pos = set(int(i) for i in self.info_positions)
        position = [0, 0, n, N]

        while not (bit_matrix[n][phi] == 0 or bit_matrix[n][phi] == 1):
            up_llr = llr_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
            up_bit = bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
            left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
            left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)]
            right_llr = llr_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
            ]
            right_bit = bit_matrix[position[0] + 1][
                position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
            ]

            if _all_num(up_bit):
                position = _up(position)
            elif _all_num(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
            elif _all_num(right_llr):
                if position[0] == position[2] - 1:
                    right_bit_pos = position[1] + 1
                    right_bit = 0 if right_llr[0] >= 0 else 1 if right_bit_pos in information_pos else 0
                    bit_matrix[position[0] + 1][
                        position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                    ] = right_bit
                else:
                    position = _rightdown(position)
            elif _all_num(left_bit):
                length = left_bit.size
                right_llr = np.array(
                    [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)],
                    dtype=np.float64,
                )
                llr_matrix[position[0] + 1][
                    position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0])
                ] = right_llr
            elif not _all_num(left_llr):
                left_llr = f_operation(up_llr[: up_llr.size // 2], up_llr[up_llr.size // 2 :])
                llr_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
            else:
                if position[0] == position[2] - 1:
                    left_bit_pos = position[1]
                    left_bit = 0 if left_llr[0] >= 0 else 1 if left_bit_pos in information_pos else 0
                    bit_matrix[position[0] + 1][position[1] : position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
                else:
                    position = _leftdown(position)

        leaf_llr = llr_matrix[n][phi]
        u_partial = bit_matrix[n].copy()
        u_partial[np.isnan(u_partial)] = 0
        return u_partial.astype(int), float(leaf_llr), llr_matrix, bit_matrix
