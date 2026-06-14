"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _info_from_frozen, _prepare_llr, sc_decode
from sc_tree_ops import pm_update_hf, sc_step_to_position, sc_tree_decode


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


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
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


def _init_matrices(y_llr, n, N):
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        y_llr = _prepare_llr(llr_ch)
        info = _info_from_frozen(self.frozen_bits)
        N, n = self.N, self.n

        llr_paths, bit_paths, pm_paths = [], [], []
        llr0, bit0 = _init_matrices(y_llr, n, N)
        llr_paths.append(llr0)
        bit_paths.append(bit0)
        pm_paths.append(0.0)

        split_positions = list(info) + ([N - 1] if (N - 1) not in info else [])
        prev = -1
        for split_pos in split_positions:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_paths, bit_paths, pm_paths):
                llr_m, bit_m = sc_step_to_position(
                    llr_m.copy(), bit_m.copy(), info, 0, split_pos
                )
                start = prev + 1
                llr_seg = llr_m[n][start : split_pos + 1]
                bit_seg = bit_m[n][start : split_pos + 1]

                pm_add = pm_update_hf(llr_seg, bit_seg)
                new_llr.append(llr_m)
                new_bit.append(bit_m)
                new_pm.append(pm + pm_add)

                if split_pos in info:
                    alt = bit_m.copy()
                    alt[n][split_pos] = 1 - alt[n][split_pos]
                    wrong_seg = alt[n][start : split_pos + 1]
                    pm_wrong = pm + pm_update_hf(llr_seg, wrong_seg)
                    new_llr.append(llr_m.copy())
                    new_bit.append(alt)
                    new_pm.append(pm_wrong)

            order = np.argsort(new_pm)[: self.list_size]
            llr_paths = [new_llr[i] for i in order]
            bit_paths = [new_bit[i] for i in order]
            pm_paths = [new_pm[i] for i in order]
            prev = split_pos

        if bit_paths[0][n][N - 1] != 0 and bit_paths[0][n][N - 1] != 1:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_paths, bit_paths, pm_paths):
                llr_m, bit_m = sc_step_to_position(
                    llr_m.copy(), bit_m.copy(), info, 0, N - 1
                )
                start = prev + 1
                pm_add = pm_update_hf(llr_m[n][start:N], bit_m[n][start:N])
                new_llr.append(llr_m)
                new_bit.append(bit_m)
                new_pm.append(pm + pm_add)
            order = np.argsort(new_pm)[: self.list_size]
            llr_paths = [new_llr[i] for i in order]
            bit_paths = [new_bit[i] for i in order]
            pm_paths = [new_pm[i] for i in order]

        candidates = []
        for bm, pm in zip(bit_paths, pm_paths):
            u_hat = np.zeros(N, dtype=int)
            for i, v in enumerate(bm[n]):
                u_hat[i] = 0 if v == 0 else 1
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda x: x[0]) if valid else min(
                candidates, key=lambda x: x[0]
            )
        else:
            best = min(candidates, key=lambda x: x[0])

        return best[1], best[0]
