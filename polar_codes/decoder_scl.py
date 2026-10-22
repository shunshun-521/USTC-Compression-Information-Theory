"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import cn_boxplus, g_operation, sc_decode_recursive, LLR_MAX


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


def _full_decode_u_up(llr, frozen, offset, u_hat, phi):
    """在已知 u_hat[0:phi] 时，完整译码子树并返回 (u_blk, u_up)。"""
    n = len(llr)
    if n == 1:
        idx = offset
        if frozen[0]:
            bit = 0
        elif idx < phi:
            bit = int(u_hat[idx])
        else:
            bit = 0 if llr[0] >= 0 else 1
        u = np.array([bit], dtype=int)
        return u, u

    half = n // 2
    llr_u = cn_boxplus(llr[:half], llr[half:])
    u1, u1_up = _full_decode_u_up(llr_u, frozen[:half], offset, u_hat, phi)
    llr_v = g_operation(llr[:half], llr[half:], u1_up)
    u2, u2_up = _full_decode_u_up(llr_v, frozen[half:], offset + half, u_hat, phi)
    u_hat_blk = np.concatenate([u1, u2])
    u1_up_xor = (u1_up.astype(int) ^ u2_up.astype(int)) % 2
    u_up = np.concatenate([u1_up_xor, u2_up])
    return u_hat_blk, u_up


def _sc_bit_llr(llr_ch, frozen_bits, u_hat, phi):
    """计算位 phi 的 LLR（前缀 u_hat[0:phi] 已固定）。"""

    def rec(llr, fr, offset):
        n = len(llr)
        if n == 1:
            return float(llr[0])
        half = n // 2
        if phi < offset + half:
            return rec(cn_boxplus(llr[:half], llr[half:]), fr[:half], offset)
        llr_u = cn_boxplus(llr[:half], llr[half:])
        _, u1_up = _full_decode_u_up(llr_u, fr[:half], offset, u_hat, phi)
        llr_v = g_operation(llr[:half], llr[half:], u1_up)
        return rec(llr_v, fr[half:], offset + half)

    return rec(llr_ch, frozen_bits, 0)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return pm if u_bit == hard else pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -LLR_MAX, LLR_MAX)

        if self.list_size == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for p in paths:
                llr_bit = _sc_bit_llr(llr_ch, self.frozen_bits, p["u"], phi)
                if self.frozen_bits[phi]:
                    q = {"pm": self._pm_update(p["pm"], llr_bit, 0), "u": p["u"].copy()}
                    q["u"][phi] = 0
                    new_paths.append(q)
                else:
                    for bit in (0, 1):
                        q = {"pm": self._pm_update(p["pm"], llr_bit, bit), "u": p["u"].copy()}
                        q["u"][phi] = bit
                        new_paths.append(q)
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda x: x["pm"]) if valid else paths[0]
        else:
            best = paths[0]
        return best["u"], best["pm"]
