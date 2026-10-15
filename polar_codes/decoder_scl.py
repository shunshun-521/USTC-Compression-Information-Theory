"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive, _frozen_to_bool


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位"""
    poly = _crc_poly(crc_length)
    reg = 0
    info_bits = np.asarray(info_bits, dtype=int)
    steps = crc_length
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(steps):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _next_bit_llr(llr, frozen, u_prefix, phi):
    """已知 u[0:phi] 时计算比特 phi 的 LLR"""
    result = [0.0]
    frozen = _frozen_to_bool(frozen)

    def decode(lam, lane, size):
        if size == 1:
            if lane == phi:
                result[0] = float(lam[0])
                bit = 0
            elif lane < phi:
                bit = int(u_prefix[lane])
            else:
                bit = 0
            return np.array([bit], dtype=int)

        half = size // 2
        left_lam = f_operation(lam[:half], lam[half:])
        left_s = decode(left_lam, lane, half)
        right_lam = g_operation(lam[:half], lam[half:], left_s)
        right_s = decode(right_lam, lane + half, half)
        s = np.zeros(size, dtype=int)
        s[:half] = left_s ^ right_s
        s[half:] = right_s
        return s

    decode(np.asarray(llr, dtype=np.float64), 0, len(llr))
    return result[0]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径仅保存 u 前缀与 PM）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = _frozen_to_bool(frozen_bits)
        self.L = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]

    @staticmethod
    def _path_metric(pm, llr, u):
        if u == 0:
            return pm + (0.0 if llr >= 0 else abs(llr))
        return pm + (0.0 if llr < 0 else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr0 = _next_bit_llr(llr_ch, self.frozen, path["u"], phi)
                if self.frozen[phi]:
                    pm = self._path_metric(path["pm"], llr0, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u_bit in (0, 1):
                        pm = self._path_metric(path["pm"], llr0, u_bit)
                        candidates.append((pm, path, u_bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.L]

            new_paths = []
            for pm, parent, u_bit in candidates:
                u_new = parent["u"].copy()
                if self.frozen[phi]:
                    u_bit = 0
                u_new[phi] = u_bit
                new_paths.append({"pm": pm, "u": u_new})
            paths = new_paths

        paths.sort(key=lambda p: p["pm"])

        if self.crc_length > 0:
            for path in paths:
                payload = path["u"][self.info_indices]
                if crc_check(payload, self.crc_length):
                    return path["u"], path["pm"]

        best = paths[0]
        return best["u"], best["pm"]
