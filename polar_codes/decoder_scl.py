"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    path_metric_penalty,
)
from encoder import bit_reversed_index


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（aff3ct 风格 LFSR）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0

    for bit in info_bits:
        fb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if int(bit) ^ fb:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0

    for bit in bits:
        fb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if int(bit) ^ fb:
            reg ^= poly

    return reg == 0


class SCLDecoder:
    """SCL 译码器（Permuted SCD）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [
            {
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, self.n)
                llr_val = L[l, self.n]

                if l in self.frozen_set:
                    new_path = {
                        "L": L.copy(),
                        "B": B.copy(),
                        "pm": path["pm"] + path_metric_penalty(llr_val, 0),
                    }
                    new_path["B"][l, self.n] = 0
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = {
                            "L": L.copy(),
                            "B": B.copy(),
                            "pm": path["pm"] + path_metric_penalty(llr_val, u_bit),
                        }
                        new_path["B"][l, self.n] = u_bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

            for path in paths:
                _update_bits(path["B"], l, self.n, self.N)

        paths.sort(key=lambda p: p["pm"])

        if self.crc_length > 0:
            for path in paths:
                u_hat = path["B"][:, self.n].astype(int)
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    return u_hat, path["pm"]

        best = paths[0]
        return best["B"][:, self.n].astype(int), best["pm"]
