"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    info_bits = np.asarray(info_bits, dtype=int)
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _decode_subtree_fixed(llrs, frozen_ind, bit_start, u_hat):
    """在子树内按已知 u_hat 重算中间比特（用于 g 运算）。"""
    n = len(llrs)
    if n == 1:
        idx = bit_start
        u = 0 if frozen_ind[0] else int(u_hat[idx])
        return np.array([u], dtype=int), np.array([u], dtype=int)

    half = n // 2
    llr1, llr2 = llrs[:half], llrs[half:]
    f1, f2 = frozen_ind[:half], frozen_ind[half:]
    mid = bit_start + half

    x_llr1 = f_operation(llr1, llr2)
    u1, u1_up = _decode_subtree_fixed(x_llr1, f1, bit_start, u_hat)
    x_llr2 = g_operation(llr1, llr2, u1_up)
    u2, u2_up = _decode_subtree_fixed(x_llr2, f2, mid, u_hat)

    u = np.concatenate([u1, u2])
    u1_up = (u1_up.astype(int) ^ u2_up.astype(int)).astype(int)
    u_up = np.concatenate([u1_up, u2_up])
    return u, u_up


def _bit_llr(llr_ch, frozen_ind, u_hat, phi):
    """计算第 phi 个比特的 LLR（前缀 u_hat[0:phi] 已确定）。"""

    def rec(llrs, f_ind, bit_start):
        n = len(llrs)
        if n == 1:
            return llrs[0]
        half = n // 2
        llr1, llr2 = llrs[:half], llrs[half:]
        f1, f2 = f_ind[:half], f_ind[half:]
        mid = bit_start + half

        x_llr1 = f_operation(llr1, llr2)
        if phi < mid:
            return rec(x_llr1, f1, bit_start)
        _, u1_up = _decode_subtree_fixed(x_llr1, f1, bit_start, u_hat)
        x_llr2 = g_operation(llr1, llr2, u1_up)
        return rec(x_llr2, f2, mid)

    return rec(llr_ch, frozen_ind, 0)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_ind = self.frozen_bits.astype(int)
        self.list_size = list_size
        self.crc_length = crc_length

    @staticmethod
    def _pm_update(pm, llr_val, u_bit):
        consistent = (u_bit == 0 and llr_val >= 0) or (u_bit == 1 and llr_val < 0)
        return pm + (0.0 if consistent else abs(llr_val))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_phi = _bit_llr(llr_ch, self.frozen_ind, path["u"], phi)
                if self.frozen_bits[phi]:
                    u_new = path["u"].copy()
                    u_new[phi] = 0
                    pm = self._pm_update(path["pm"], llr_phi, 0)
                    new_paths.append({"pm": pm, "u": u_new})
                else:
                    for bit in (0, 1):
                        u_new = path["u"].copy()
                        u_new[phi] = bit
                        pm = self._pm_update(path["pm"], llr_phi, bit)
                        new_paths.append({"pm": pm, "u": u_new})
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            if valid:
                paths = valid
        best = min(paths, key=lambda x: x["pm"])
        return best["u"], best["pm"]
