"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, _prepare_llr
from utils import crc_check_bits


def crc_encode(info_bits, crc_length=8):
    from utils import crc_encode_bits

    return crc_encode_bits(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    return crc_check_bits(bits, crc_length)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        if u_bit != hard:
            pm += abs(llr_val)
        return pm

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        paths = [(0.0, llr, np.array([], dtype=int), np.array([], dtype=int))]

        def extend_paths(llr_block, frozen_block, paths_in):
            n = len(llr_block)
            if n == 1:
                new_paths = []
                llr_val = llr_block[0]
                for pm, _, u_hat, u_up in paths_in:
                    if frozen_block[0]:
                        new_paths.append((self._pm_update(pm, llr_val, 0), None, np.array([0]), np.array([0])))
                    else:
                        for bit in (0, 1):
                            new_paths.append(
                                (
                                    self._pm_update(pm, llr_val, bit),
                                    None,
                                    np.array([bit]),
                                    np.array([bit]),
                                )
                            )
                new_paths.sort(key=lambda x: x[0])
                return new_paths[: self.list_size]

            half = n // 2
            llr_left = f_operation(llr_block[:half], llr_block[half:])
            left_paths = extend_paths(llr_left, frozen_block[:half], paths_in)

            right_candidates = []
            for pm, _, u_left, u_left_up in left_paths:
                llr_right = g_operation(llr_block[:half], llr_block[half:], u_left_up)
                suffix_paths = extend_paths(
                    llr_right,
                    frozen_block[half:],
                    [(pm, None, np.array([]), np.array([]))],
                )
                for spm, _, u_right, u_right_up in suffix_paths:
                    pm_total = pm + (spm - 0.0)
                    u_hat = np.concatenate([u_left, u_right])
                    u_left_xor = np.bitwise_xor(u_left_up, u_right_up)
                    u_up = np.concatenate([u_left_xor, u_right_up])
                    right_candidates.append((pm_total, None, u_hat, u_up))

            right_candidates.sort(key=lambda x: x[0])
            return right_candidates[: self.list_size]

        paths = extend_paths(llr, self.frozen_bits, paths)

        if self.crc_length > 0:
            valid = []
            for pm, _, u_hat, _ in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check_bits(info_bits, self.crc_length):
                    valid.append((pm, _, u_hat, _))
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x[0])
        return best[2], best[0]
