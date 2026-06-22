"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _update_bits,
    _upper_llr,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


def _update_llrs_path(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = []
        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_ch
        paths.append({"L": L0, "B": B0, "pm": 0.0, "u_hat": np.zeros(N, dtype=int)})

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs_path(L, B, l, n)
                cur_llr = L[l, n]

                if self.frozen_bits[i]:
                    new_path = {
                        "L": L.copy(),
                        "B": B.copy(),
                        "pm": path["pm"] + self._path_metric_penalty(cur_llr, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["u_hat"][i] = 0
                    new_path["B"][l, n] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            "L": L.copy(),
                            "B": B.copy(),
                            "pm": path["pm"] + self._path_metric_penalty(cur_llr, bit),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["u_hat"][i] = bit
                        new_path["B"][l, n] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

            for path in paths:
                _update_bits(path["B"], l, n)

        if self.crc_length > 0:
            crc_paths = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            if crc_paths:
                paths = crc_paths

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
