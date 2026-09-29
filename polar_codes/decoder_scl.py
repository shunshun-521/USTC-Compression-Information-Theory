"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode, sc_decode_recursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _bit_llr(llr, u_prefix, phi, frozen_bits):
    """计算第 phi 个比特的 LLR（已知前缀 u_prefix[:phi]）"""
    n = int(np.log2(len(llr)))

    def recur_llr(L, fz, depth, offset, target):
        N = 1 << depth
        if depth == 0:
            return L[0]
        half = N // 2
        L1 = L[:half]
        L2 = L[half:]
        fz1 = fz[:half]
        fz2 = fz[half:]

        if target < half:
            x_llr1 = f_operation(L1, L2)
            return recur_llr(x_llr1, fz1, depth - 1, offset, target)
        u_left = u_prefix[offset : offset + half]
        u_hat1_up = _partial_encode(u_left, depth - 1)
        x_llr2 = g_operation(L1, L2, u_hat1_up)
        return recur_llr(x_llr2, fz2, depth - 1, offset + half, target - half)

    return recur_llr(llr, frozen_bits.astype(float), n, 0, phi)


def _partial_encode(u_seg, depth):
    """段内极化变换后的部分编码比特（用于 g 运算）"""
    if depth == 0:
        return u_seg.astype(np.float64)
    half = 1 << (depth - 1)
    u1 = u_seg[:half]
    u2 = u_seg[half:]
    up1 = _partial_encode(u1, depth - 1)
    up2 = _partial_encode(u2, depth - 1)
    up1_i = (up1.astype(np.int8) ^ up2.astype(np.int8)).astype(np.float64)
    return np.concatenate([up1_i, up2])


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        N = self.N
        paths = [{"pm": 0.0, "u": np.zeros(N, dtype=int)}]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr_b = _bit_llr(llr_ch, path["u"], phi, self.frozen_bits)
                if self.frozen_bits[phi]:
                    pm = self._pm_update(path["pm"], llr_b, 0)
                    u = path["u"].copy()
                    u[phi] = 0
                    new_paths.append({"pm": pm, "u": u})
                else:
                    for bit in (0, 1):
                        pm = self._pm_update(path["pm"], llr_b, bit)
                        u = path["u"].copy()
                        u[phi] = bit
                        new_paths.append({"pm": pm, "u": u})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info = p["u"][self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        return best["u"], best["pm"]
