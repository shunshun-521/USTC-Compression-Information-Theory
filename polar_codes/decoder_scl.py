"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _as_frozen_mask,
    _polar_decode_sc_core,
    sc_decode_recursive,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


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
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if int(u) != hard:
        return pm + abs(llr)
    return pm


def _polar_decode_sc_forced(llr_ch, frozen_ind, u_ref, phi_limit, offset):
    """在已知 u_ref[0:phi_limit) 时递归计算 partial-sum。"""
    n = len(frozen_ind)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    if n > 1:
        half = n // 2
        l1 = llr_ch[:half]
        l2 = llr_ch[half:]
        f1 = frozen_ind[:half]
        f2 = frozen_ind[half:]
        llr_left = f_operation(l1, l2)
        u_hat1, u_hat1_up = _polar_decode_sc_forced(llr_left, f1, u_ref, phi_limit, offset)
        llr_right = g_operation(l1, l2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_forced(llr_right, f2, u_ref, phi_limit, offset + half)
        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_i = (u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)).astype(np.float64)
        u_hat_up = np.concatenate([u_hat1_up_i, u_hat2_up])
        return u_hat, u_hat_up

    idx = offset
    if idx < phi_limit:
        u_val = float(u_ref[idx])
    elif frozen_ind[0]:
        u_val = 0.0
    else:
        u_val = 0.0 if llr_ch[0] >= 0 else 1.0
    return np.array([u_val]), np.array([u_val])


def _llr_at_phase(phi, llr_rev, frozen_ind, u_prefix):
    """计算路径前缀 u_prefix[0:phi] 下第 phi 个比特的 SC LLR。"""

    def rec(llr_seg, f_seg, offset):
        m = len(f_seg)
        if m == 1:
            return float(llr_seg[0])
        half = m // 2
        l1 = llr_seg[:half]
        l2 = llr_seg[half:]
        f1 = f_seg[:half]
        f2 = f_seg[half:]
        if phi < offset + half:
            return rec(f_operation(l1, l2), f1, offset)
        llr_left = f_operation(l1, l2)
        _, u_up_left = _polar_decode_sc_forced(llr_left, f1, u_prefix, phi, offset)
        llr_right = g_operation(l1, l2, u_up_left)
        return rec(llr_right, f2, offset + half)

    return rec(llr_rev, frozen_ind, 0)


class SCLDecoder:
    """SCL 译码器（路径列表 + CRC 辅助）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = _as_frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_ind = self.frozen_bits.astype(np.float64)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_rev = llr_ch[rev]

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=np.int8)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = _llr_at_phase(phi, llr_rev, self.frozen_ind, path["u"])
                if self.frozen_bits[phi]:
                    pm = _pm_update(path["pm"], llr, 0)
                    u = path["u"].copy()
                    u[phi] = 0
                    candidates.append((pm, u))
                else:
                    for bit in (0, 1):
                        pm = _pm_update(path["pm"], llr, bit)
                        u = path["u"].copy()
                        u[phi] = bit
                        candidates.append((pm, u))

            candidates.sort(key=lambda x: x[0])
            paths = [{"pm": pm, "u": u} for pm, u in candidates[: self.list_size]]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
