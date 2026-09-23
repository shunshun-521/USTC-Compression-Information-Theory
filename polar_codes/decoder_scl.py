"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) ^ (poly if feedback else 0)
    for _ in range(crc_length):
        feedback = ((reg >> (crc_length - 1)) & 1)
        reg = ((reg << 1) & mask) ^ (poly if feedback else 0)
    return reg


_CRC_POLY = {8: 0xE0, 16: 0xA001}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC_POLY[crc_length]
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC_POLY[crc_length]
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _penalty(self, llr, bit):
        hard = 1 if llr < 0 else 0
        return 0.0 if hard == bit else abs(llr)

    def _list_decode(self, llr, depth, node, paths):
        if depth == self.n - 1:
            new_paths = []
            for path in paths:
                if node in self.frozen_set:
                    path["pm"] += self._penalty(llr[0], 0)
                    path["bits"][node] = 0
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        cp = {"pm": path["pm"] + self._penalty(llr[0], bit), "bits": path["bits"].copy()}
                        cp["bits"][node] = bit
                        new_paths.append(cp)
            new_paths.sort(key=lambda p: p["pm"])
            return new_paths[: self.list_size]

        half = len(llr) // 2
        left = f_operation(np.array(llr[:half]), np.array(llr[half:]))
        left_paths = self._list_decode(left.tolist(), depth + 1, 2 * node, paths)

        all_paths = []
        for lp in left_paths:
            u_left = lp["bits"][2 * node] if 2 * node < self.N else 0
            right_llr = g_operation(
                np.array(llr[:half]), np.array(llr[half:]), [u_left]
            ).tolist()
            right_paths = self._list_decode(right_llr, depth + 1, 2 * node + 1, [lp])
            all_paths.extend(right_paths)

        all_paths.sort(key=lambda p: p["pm"])
        return all_paths[: self.list_size]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        init = [{"pm": 0.0, "bits": np.zeros(self.N, dtype=np.int8)}]
        paths = self._list_decode(llr_ch.tolist(), 0, 0, init)

        if not paths:
            return np.zeros(self.N, dtype=int), 0.0

        best_pm = float("inf")
        best_bits = paths[0]["bits"]
        crc_valid = []

        for path in paths:
            if self.crc_length > 0:
                info_bits = path["bits"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)
            if path["pm"] < best_pm:
                best_pm = path["pm"]
                best_bits = path["bits"]

        if self.crc_length > 0 and crc_valid:
            best = min(crc_valid, key=lambda p: p["pm"])
            best_bits = best["bits"]
            best_pm = best["pm"]

        return best_bits.astype(int), best_pm
