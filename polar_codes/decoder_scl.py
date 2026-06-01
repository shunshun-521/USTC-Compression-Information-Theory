"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _path_penalty(llr, u_bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)


def _compute_u_up(u, offset, n):
    """计算 Sionna 风格 u_hat_up。"""
    if n == 1:
        return np.array([float(u[offset])])
    half = n // 2
    left_up = _compute_u_up(u, offset, half)
    right = u[offset + half : offset + n].astype(np.float64)
    left_xor = np.bitwise_xor(left_up.astype(int), right.astype(int)).astype(np.float64)
    return np.concatenate([left_xor, right])


def _scl_rec(llr, frozen, offset, u, list_size):
    """递归 SCL，返回路径列表 [{'u': ndarray, 'pm': float}, ...]。"""
    n = len(llr)
    if n == 1:
        idx = offset
        llr0 = float(llr[0])
        out = []
        if frozen[0]:
            u_copy = u.copy()
            u_copy[idx] = 0
            out.append({"u": u_copy, "pm": _path_penalty(llr0, 0)})
        else:
            for bit in (0, 1):
                u_copy = u.copy()
                u_copy[idx] = bit
                out.append({"u": u_copy, "pm": _path_penalty(llr0, bit)})
        return out

    half = n // 2
    llr1, llr2 = llr[:half], llr[half:]
    f1, f2 = frozen[:half], frozen[half:]

    left_paths = _scl_rec(f_operation(llr1, llr2), f1, offset, u, list_size)

    merged = []
    for lp in left_paths:
        u_up = _compute_u_up(lp["u"], offset, half)
        llr_r = g_operation(llr1, llr2, u_up)
        right_paths = _scl_rec(llr_r, f2, offset + half, lp["u"], list_size)
        for rp in right_paths:
            merged.append({"u": rp["u"], "pm": lp["pm"] + rp["pm"]})

    merged.sort(key=lambda p: p["pm"])
    return merged[:list_size]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """SCL 译码，返回 (u_hat, pm)。"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.rev]
        u0 = np.zeros(self.N, dtype=int)

        paths = _scl_rec(llr, self.frozen_bits, 0, u0, self.list_size)

        for p in paths:
            p["u"][self.frozen_bits] = 0

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p["pm"])
        return best["u"], best["pm"]
