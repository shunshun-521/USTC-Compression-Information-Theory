"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import cn_op, g_operation, sc_decode

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_penalty(llr, u):
    preferred = 0 if llr >= 0 else 1
    return 0.0 if u == preferred else abs(llr)


def _scl_decode_core(llr_ch, frozen_ind, list_size):
    """递归 SCL 核心，返回路径列表 [(pm, u_hat, u_up), ...]"""
    n = len(llr_ch)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)

    if n == 1:
        paths = []
        llr = float(llr_ch[0])
        if frozen_ind[0] == 1:
            paths.append(( _pm_penalty(llr, 0), np.array([0]), np.array([0.0]) ))
        else:
            for u in (0, 1):
                paths.append(( _pm_penalty(llr, u), np.array([u]), np.array([float(u)]) ))
        paths.sort(key=lambda x: x[0])
        return paths[:list_size]

    half = n // 2
    llr_left = cn_op(llr_ch[:half], llr_ch[half:])
    left_paths = _scl_decode_core(llr_left, frozen_ind[:half], list_size)

    merged = []
    for pm_l, u_l, up_l in left_paths:
        llr_right = g_operation(llr_ch[:half], llr_ch[half:], up_l)
        right_paths = _scl_decode_core(llr_right, frozen_ind[half:], list_size)
        for pm_r, u_r, up_r in right_paths:
            pm = pm_l + pm_r
            u_hat = np.concatenate([u_l, u_r])
            up_left = np.bitwise_xor(up_l.astype(int), up_r.astype(int)).astype(np.float64)
            u_up = np.concatenate([up_left, up_r])
            merged.append((pm, u_hat, u_up))

    merged.sort(key=lambda x: x[0])
    return merged[:list_size]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_ind = np.zeros(N, dtype=np.float64)
        self.frozen_ind[self.frozen_bits] = 1.0
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = _scl_decode_core(llr_ch, self.frozen_ind, self.list_size)

        best_crc = None
        if self.crc_length > 0:
            for pm, u_hat, _ in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or pm < best_crc[0]:
                        best_crc = (pm, u_hat)

        if best_crc is not None:
            return best_crc[1].astype(int), best_crc[0]

        pm, u_hat, _ = paths[0]
        return u_hat.astype(int), pm
