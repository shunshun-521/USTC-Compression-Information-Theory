"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _bit_reversed, _update_bits, _update_llrs


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            is_frozen = self.frozen_bits[l]
            candidates = []

            for pidx, path in enumerate(paths):
                L_tmp = path["L"].copy()
                B_tmp = path["B"].copy()
                _update_llrs(L_tmp, B_tmp, l, n)
                llr = L_tmp[l, n]

                if is_frozen:
                    candidates.append((
                        path["pm"] + self._pm_penalty(llr, 0),
                        pidx, 0, L_tmp.copy(), B_tmp.copy(),
                    ))
                else:
                    for bit in (0, 1):
                        candidates.append((
                            path["pm"] + self._pm_penalty(llr, bit),
                            pidx, bit, L_tmp.copy(), B_tmp.copy(),
                        ))

            candidates.sort(key=lambda x: x[0])
            new_paths = []

            for pm, pidx, bit, L_tmp, B_tmp in candidates[: self.list_size]:
                dst = {
                    "L": L_tmp,
                    "B": B_tmp,
                    "pm": pm,
                }
                dst["B"][l, n] = bit
                _update_bits(dst["B"], l, n)
                new_paths.append(dst)

            paths = new_paths

        crc_valid = []
        best_u, best_pm = None, np.inf
        for path in paths:
            u = path["B"][:, n].astype(int).copy()
            if self.crc_length > 0:
                info_part = u[self.info_indices]
                if crc_check(info_part, self.crc_length):
                    crc_valid.append((path["pm"], u))
            if path["pm"] < best_pm:
                best_pm = path["pm"]
                best_u = u

        if self.crc_length > 0 and crc_valid:
            crc_valid.sort(key=lambda x: x[0])
            return crc_valid[0][1], crc_valid[0][0]

        return best_u, best_pm
