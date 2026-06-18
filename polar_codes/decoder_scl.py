"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _info_indices, _prepare_llr, sc_decode
from encoder import bit_reversal_permutation
from _ref_decoder import scl_decoder as _ref_scl_decoder


from _ref_crc import CRC


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    coded = CRC(info_bits.tolist(), crc_length).code
    return np.array(coded, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits_list = list(np.asarray(bits, dtype=int))
    obj = CRC(bits_list[: len(bits_list) - crc_length], crc_length)
    obj.info = bits_list
    return obj.detection() == 1


class SCLDecoder:
    """SCL 译码器（基于参考实现的列表译码）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = _info_indices(frozen_bits)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        y_llr = _prepare_llr(llr_ch)
        info_pos = list(self.info_indices)

        if self.crc_length == 0:
            u_hat = self._scl_no_crc(y_llr, info_pos)
        else:
            u_hat = self._scl_with_crc(y_llr, info_pos)

        return u_hat.astype(int), 0.0

    def _scl_no_crc(self, y_llr, info_pos):
        """无 CRC 的 SCL：调用参考实现并选取最小路径度量。"""
        from _ref_decoder import sc_stepping_decoder
        import _ref_function as function

        N = self.N
        n = self.n
        list_max = self.list_size
        pm_method = "hf"

        llr_matrix = np.ones((n + 1, N))
        llr_matrix[llr_matrix == 1] = float("nan")
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = y_llr

        llr_list = [llr_matrix]
        bit_list = [bit_matrix]
        pm_list = [0.0]
        split_pos = info_pos
        split_loc = 0
        split_len = len(split_pos)
        l_now = 1

        while split_len - 1 >= split_loc:
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []
            for i in range(l_now):
                matrix_temp = sc_stepping_decoder(
                    llr_list[i].copy(),
                    bit_list[i].copy(),
                    info_pos,
                    0,
                    split_pos[split_loc],
                )
                pm_base = pm_list[i]
                prev_end = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                cur_end = split_pos[split_loc] + 1

                right_pm = function.get_pm_update(
                    matrix_temp[0][n][prev_end:cur_end],
                    matrix_temp[1][n][prev_end:cur_end],
                    pm_method,
                )
                new_llr_list.append(matrix_temp[0])
                new_bit_list.append(matrix_temp[1])
                new_pm_list.append(pm_base + right_pm)

                wrong_bit = matrix_temp[1].copy()
                wrong_bit[n][split_pos[split_loc]] = (
                    1 - wrong_bit[n][split_pos[split_loc]]
                )
                wrong_pm = function.get_pm_update(
                    matrix_temp[0][n][prev_end:cur_end],
                    wrong_bit[n][prev_end:cur_end],
                    pm_method,
                )
                new_llr_list.append(matrix_temp[0].copy())
                new_bit_list.append(wrong_bit)
                new_pm_list.append(pm_base + wrong_pm)

            if len(new_pm_list) > list_max:
                keep = np.argsort(new_pm_list)[:list_max]
                new_pm_list = [new_pm_list[i] for i in keep]
                new_llr_list = [new_llr_list[i] for i in keep]
                new_bit_list = [new_bit_list[i] for i in keep]

            llr_list, bit_list, pm_list = new_llr_list, new_bit_list, new_pm_list
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for i in range(l_now):
                prev_end = split_pos[split_loc - 1] + 1
                matrix_temp = sc_stepping_decoder(
                    llr_list[i], bit_list[i], info_pos, 0, N - 1
                )
                pm_update = function.get_pm_update(
                    matrix_temp[0][n][prev_end:N],
                    matrix_temp[1][n][prev_end:N],
                    pm_method,
                )
                llr_list[i] = matrix_temp[0]
                bit_list[i] = matrix_temp[1]
                pm_list[i] += pm_update

        best_idx = int(np.argmin(pm_list))
        u_d = bit_list[best_idx][n]
        return np.array([0 if v == 0 else 1 for v in u_d], dtype=int)

    def _scl_with_crc(self, y_llr, info_pos):
        """CRC 辅助 SCL。"""
        u_hat = _ref_scl_decoder(
            y_llr, info_pos, 0, [self.list_size, "hf"], self.crc_length
        )
        if isinstance(u_hat, np.ndarray):
            return u_hat.astype(int)
        return np.asarray(u_hat, dtype=int)
