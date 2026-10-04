"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, sc_decode


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
        width = 8
    elif crc_length == 16:
        poly = 0x8005
        width = 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    mask = (1 << width) - 1
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        if reg & (1 << (width - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（基于 SC 因子图逐比特扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.LARGE = 1e6

    def _path_metric_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return abs(llr) if u != hard else 0.0

    def _bit_llr(self, llr_ch, R0, phi):
        """给定左向先验 R0，计算比特 phi 的 LLR（单遍 BP 边际）"""
        N = self.N
        n = self.n
        L = np.zeros((N, n + 1), dtype=np.float64)
        R = np.zeros((N, n + 1), dtype=np.float64)
        L[:, n] = llr_ch
        R[:, 0] = R0.copy()
        R[self.frozen_bits == 1, 0] = self.LARGE
        alpha = 0.9375

        for j in range(n, 0, -1):
            s = 1 << (j - 1)
            for i in range(0, N, 2 * s):
                for t in range(s):
                    idx = i + t
                    idx2 = i + t + s
                    L[idx, j - 1] = alpha * f_operation(R[idx, j] + L[idx2, j], L[idx, j])
                    L[idx2, j - 1] = alpha * f_operation(R[idx, j], L[idx, j]) + L[idx2, j]
        return L[phi, 0] + R[phi, 0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0
        N = self.N
        n = self.n
        paths = [{"pm": 0.0, "u": np.zeros(N, dtype=int), "R0": np.zeros(N, dtype=np.float64)}]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr = self._bit_llr(llr_ch, path["R0"], phi)
                if self.frozen_bits[phi]:
                    p = {
                        "pm": path["pm"] + self._path_metric_penalty(llr, 0),
                        "u": path["u"].copy(),
                        "R0": path["R0"].copy(),
                    }
                    p["u"][phi] = 0
                    p["R0"][phi] = self.LARGE
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = {
                            "pm": path["pm"] + self._path_metric_penalty(llr, bit),
                            "u": path["u"].copy(),
                            "R0": path["R0"].copy(),
                        }
                        p["u"][phi] = bit
                        p["R0"][phi] = self.LARGE if bit == 0 else -self.LARGE
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_paths = []
        if self.crc_length > 0:
            for p in paths:
                info_bits = p["u"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(p)
        candidates = crc_paths if crc_paths else paths
        best = min(candidates, key=lambda p: p["pm"])
        return best["u"].copy(), best["pm"]
