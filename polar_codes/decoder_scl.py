"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        if reg & (1 << (degree - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
        else:
            reg = (reg << 1) & ((1 << degree) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly, degree = CRC8_POLY, 8
    elif crc_length == 16:
        poly, degree = CRC16_POLY, 16
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    remainder = _crc_remainder(info_bits, poly, degree)
    crc_bits = np.array(
        [(remainder >> (degree - 1 - i)) & 1 for i in range(degree)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly, degree = CRC8_POLY, 8
    elif crc_length == 16:
        poly, degree = CRC16_POLY, 16
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    return _crc_remainder(bits, poly, degree) == 0


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


def _polar_scl_rec(llr, frozen, list_size):
    """递归 SCL，返回最多 list_size 条路径 (PM, u, u_up)"""
    n = len(llr)
    frozen = np.asarray(frozen, dtype=int)

    if n == 1:
        if frozen[0]:
            return [(0.0, np.array([0]), np.array([0.0]))]
        l = float(llr[0])
        paths = []
        for u in (0, 1):
            paths.append((_pm_penalty(l, u), np.array([u]), np.array([float(u)])))
        paths.sort(key=lambda x: x[0])
        return paths[:max(1, list_size)]

    half = n // 2
    llr1, llr2 = llr[:half], llr[half:]
    f1, f2 = frozen[:half], frozen[half:]
    llr_up = f_operation(llr1, llr2)
    paths_left = _polar_scl_rec(llr_up, f1, list_size)

    all_paths = []
    for pm_l, u_l, u_up_l in paths_left:
        llr_low = g_operation(llr1, llr2, u_up_l)
        paths_right = _polar_scl_rec(llr_low, f2, list_size)
        for pm_r, u_r, u_up_r in paths_right:
            u_full = np.concatenate([u_l, u_r])
            u_up_xor = (u_up_l.astype(int) ^ u_up_r.astype(int)).astype(float)
            u_up_full = np.concatenate([u_up_xor, u_up_r])
            all_paths.append((pm_l + pm_r, u_full, u_up_full))

    all_paths.sort(key=lambda x: x[0])
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器（递归列表实现，L=1 时退化为 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = _polar_scl_rec(llr, self.frozen_bits, self.list_size)

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p[1][self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p[0])

        return best[1].astype(int), best[0]
