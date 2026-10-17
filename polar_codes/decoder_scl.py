"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _lower_llr,
    _upper_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(info_bits, dtype=int):
        reg ^= (b << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, crc_bits]).astype(int)


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    poly = _crc_poly(crc_length)
    reg = 0
    for b in np.asarray(bits, dtype=int):
        reg ^= (b << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特阵列）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen)[0])
        self.info_idx = np.where(~self.frozen)[0]

    def _path_metric_penalty(self, llr, bit):
        """与 LLR 不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.n + 1
        paths = [
            {
                "pm": 0.0,
                "L": np.full((self.N, L), np.nan, dtype=np.float64),
                "B": np.full((self.N, L), np.nan),
                "u": np.zeros(self.N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        def update_llrs(path, l):
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path["L"][j, s + 1] = _upper_llr(path["L"][j, s], path["L"][j + branch_size, s])
                    else:
                        top_bit = int(path["B"][j - branch_size, s + 1])
                        path["L"][j, s + 1] = _lower_llr(
                            path["L"][j, s], path["L"][j - branch_size, s], top_bit
                        )

        def update_bits(path, l):
            if l < self.N // 2:
                return
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path["B"][j - branch_size, s - 1] = int(path["B"][j, s]) ^ int(
                            path["B"][j - branch_size, s]
                        )
                        path["B"][j, s - 1] = path["B"][j, s]

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            new_paths = []
            for path in paths:
                update_llrs(path, l)
                llr_bit = path["L"][l, self.n]
                if l in self.frozen_set:
                    pen = self._path_metric_penalty(llr_bit, 0)
                    p2 = {
                        "pm": path["pm"] + pen,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u": path["u"].copy(),
                    }
                    p2["B"][l, self.n] = 0
                    p2["u"][l] = 0
                    update_bits(p2, l)
                    new_paths.append(p2)
                else:
                    for bit in (0, 1):
                        pen = self._path_metric_penalty(llr_bit, bit)
                        p2 = {
                            "pm": path["pm"] + pen,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u": path["u"].copy(),
                        }
                        p2["B"][l, self.n] = bit
                        p2["u"][l] = bit
                        update_bits(p2, l)
                        new_paths.append(p2)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u"][self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
