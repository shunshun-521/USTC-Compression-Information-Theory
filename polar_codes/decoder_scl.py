"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    sc_decode_factor_graph,
    sc_decode,
    _frozen_to_info_pos,
)


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return [1, 0, 0, 0, 0, 0, 1, 1, 1]
    if crc_length == 16:
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_remainder(info_bits, crc_length):
    info = list(np.asarray(info_bits, dtype=int).tolist())
    poly = _crc_polynomial(crc_length)
    msg = info + [0] * crc_length
    for i in range(len(info)):
        if msg[i] == 1:
            for j in range(len(poly)):
                msg[i + j] ^= poly[j]
    return msg[len(info) : len(info) + crc_length]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info = np.asarray(info_bits, dtype=int).tolist()
    crc_bits = _crc_remainder(info, crc_length)
    return np.array(info + crc_bits, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = _crc_remainder(info.tolist(), crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器，按比特索引逐位扩展路径。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_set = _frozen_to_info_pos(frozen_bits)
        self.info_indices = sorted(self.info_set)

    def _path_metric_add(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        br = bit_reversal_permutation(self.N)
        y_llr = np.asarray(llr_ch, dtype=np.float64)[br]

        llr_matrix = np.ones((self.n + 1, self.N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = y_llr

        paths = [(llr_matrix, bit_matrix, 0.0)]

        from decoder_sc import (
            _all_decided,
            _up,
            _leftdown,
            _rightdown,
            _get_up_bit,
            _get_right_bit,
            _get_left_bit,
            _get_right_llr,
            _get_left_llr,
        )

        for phi in range(self.N):
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m = llr_m.copy()
                bit_m = bit_m.copy()
                llr_m, bit_m, leaf_llr = self._advance_to_phi(
                    llr_m,
                    bit_m,
                    phi,
                    _all_decided,
                    _up,
                    _leftdown,
                    _rightdown,
                    _get_up_bit,
                    _get_right_bit,
                    _get_left_bit,
                    _get_right_llr,
                    _get_left_llr,
                )

                if self.frozen_bits[phi]:
                    bit_m[self.n, phi] = 0
                    pm_new = pm + self._path_metric_add(leaf_llr, 0)
                    new_paths.append((llr_m, bit_m, pm_new))
                else:
                    for bit in (0, 1):
                        bm = bit_m.copy()
                        bm[self.n, phi] = bit
                        pm_new = pm + self._path_metric_add(leaf_llr, bit)
                        new_paths.append((llr_m.copy(), bm, pm_new))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda x: x[2])

        if self.crc_length > 0:
            for _, bit_m, pm in paths:
                u_hat = bit_m[self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat, pm

        best = paths[0][1][self.n].astype(int)
        return best, paths[0][2]

    def _advance_to_phi(
        self,
        llr_matrix,
        bit_matrix,
        phi,
        _all_decided,
        _up,
        _leftdown,
        _rightdown,
        _get_up_bit,
        _get_right_bit,
        _get_left_bit,
        _get_right_llr,
        _get_left_llr,
    ):
        N = self.N
        n = self.n
        position = [0, 0, n, N]

        while not (bit_matrix[n, phi] == 0 or bit_matrix[n, phi] == 1):
            span = 2 ** (position[2] - position[0])
            up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
            up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
            half = span // 2
            left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
            left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
            right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
            right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

            if _all_decided(up_bit):
                position = _up(position)
            elif _all_decided(right_bit):
                bit_matrix[position[0]][position[1] : position[1] + span] = _get_up_bit(
                    left_bit, right_bit
                )
            elif _all_decided(right_llr):
                if position[0] == position[2] - 1:
                    val = _get_right_bit(
                        right_llr, self.info_set, 0, position[1] + 1
                    )
                    bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                        val
                    )
                else:
                    position = _rightdown(position)
            elif _all_decided(left_bit):
                llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = (
                    _get_right_llr(left_bit, up_llr)
                )
            elif not _all_decided(left_llr):
                llr_matrix[position[0] + 1][position[1] : position[1] + half] = (
                    _get_left_llr(up_llr)
                )
            else:
                if position[0] == position[2] - 1:
                    val = _get_left_bit(
                        left_llr, self.info_set, 0, position[1]
                    )
                    bit_matrix[position[0] + 1][position[1] : position[1] + half] = val
                else:
                    position = _leftdown(position)

        leaf_llr = llr_matrix[n, phi]
        if np.isnan(leaf_llr):
            leaf_llr = llr_matrix[0, phi]
        return llr_matrix, bit_matrix, leaf_llr
