"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation_exact, g_operation, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _pm_penalty(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)


def _scl_recursive(llr_ch, frozen_ind, list_size):
    """
    递归 SCL，返回最多 list_size 条路径：
    每条路径为 (u_hat_segment, pm, u_up_segment)
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    n_len = len(llr_ch)

    if n_len == 1:
        if frozen_ind[0] == 1:
            return [(np.array([0.0]), 0.0, np.array([0.0]))]
        llr = llr_ch[0]
        paths = []
        for u in (0.0, 1.0):
            paths.append((np.array([u]), _pm_penalty(llr, int(u)), np.array([u])))
        paths.sort(key=lambda x: x[1])
        return paths[:list_size]

    half = n_len // 2
    llr1 = llr_ch[:half]
    llr2 = llr_ch[half:]
    fr1 = frozen_ind[:half]
    fr2 = frozen_ind[half:]

    llr_upper = f_operation_exact(llr1, llr2)
    upper_paths = _scl_recursive(llr_upper, fr1, list_size)

    all_paths = []
    for u1, pm1, up1 in upper_paths:
        llr_lower = g_operation(llr1, llr2, up1)
        lower_paths = _scl_recursive(llr_lower, fr2, list_size)
        for u2, pm2, up2 in lower_paths:
            u_hat = np.concatenate([u1, u2])
            up1_i = (up1.astype(np.int8) ^ up2.astype(np.int8)).astype(np.float64)
            u_up = np.concatenate([up1_i, up2])
            all_paths.append((u_hat, pm1 + pm2, u_up))

    all_paths.sort(key=lambda x: x[1])
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径在递归树中分裂）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        frozen_ind = self.frozen_bits.astype(np.float64)
        paths = _scl_recursive(np.asarray(llr_ch, dtype=np.float64), frozen_ind, self.list_size)
        candidates = []
        for u_hat, pm, _ in paths:
            u_int = u_hat.astype(int)
            candidates.append((u_int, pm))

        if self.crc_length > 0:
            valid = [(u, pm) for u, pm in candidates if crc_check(u[self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda x: x[1])
            else:
                best = min(candidates, key=lambda x: x[1])
        else:
            best = min(candidates, key=lambda x: x[1])

        return best[0], best[1]
