"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

import _sc_helpers as _fn
import _sc_ref as _ref
from decoder_sc import _frozen_to_info


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) & ((1 << crc_length) - 1)) | int(bit)
        if msb ^ int(bit):
            reg ^= poly
    for _ in range(crc_length):
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器，基于已验证的顺序 SC stepping 实现。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = _frozen_to_info(frozen_bits)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        info_pos = self.information_pos
        L = self.list_size

        llr_matrix = np.ones((n + 1, N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]

        split_pos = info_pos
        split_loc = 0

        while split_loc < len(split_pos):
            new_llr, new_bit, new_pm = [], [], []
            for idx in range(len(llr_list)):
                mat = _ref.sc_stepping_decoder(
                    llr_list[idx].copy(), bit_list[idx].copy(), info_pos, 0, split_pos[split_loc]
                )
                pm_base = pm_list[idx]
                u_val = int(mat[1][n][split_pos[split_loc]])
                pm_add = _fn.get_pm_update(
                    mat[0][n][split_pos[split_loc]:split_pos[split_loc] + 1],
                    mat[1][n][split_pos[split_loc]:split_pos[split_loc] + 1],
                    "hf",
                )
                new_llr.append(mat[0])
                new_bit.append(mat[1])
                new_pm.append(pm_base + pm_add)

                if split_pos[split_loc] in info_pos:
                    wrong = mat[1].copy()
                    wrong[n][split_pos[split_loc]] = 1 - u_val
                    pm_wrong = pm_base + _fn.get_pm_update(
                        mat[0][n][split_pos[split_loc]:split_pos[split_loc] + 1],
                        wrong[n][split_pos[split_loc]:split_pos[split_loc] + 1],
                        "hf",
                    )
                    new_llr.append(mat[0].copy())
                    new_bit.append(wrong)
                    new_pm.append(pm_wrong)

            order = np.argsort(new_pm)[:L]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for idx in range(len(llr_list)):
                mat = _ref.sc_stepping_decoder(
                    llr_list[idx], bit_list[idx], info_pos, 0, N - 1
                )
                llr_list[idx] = mat[0]
                bit_list[idx] = mat[1]
                pm_list[idx] += _fn.get_pm_update(
                    mat[0][n][split_pos[-1] + 1:N],
                    mat[1][n][split_pos[-1] + 1:N],
                    "hf",
                )

        order = np.argsort(pm_list)
        best_u = None
        best_pm = None

        for idx in order:
            u_candidate = np.array([0 if v == 0 else 1 for v in bit_list[idx][n]], dtype=int)
            if self.crc_length > 0:
                payload = u_candidate[info_pos]
                if crc_check(payload, self.crc_length):
                    return u_candidate, pm_list[idx]
            elif best_u is None:
                best_u = u_candidate
                best_pm = pm_list[idx]

        if best_u is None:
            idx = order[0]
            best_u = np.array([0 if v == 0 else 1 for v in bit_list[idx][n]], dtype=int)
            best_pm = pm_list[idx]

        return best_u, best_pm
