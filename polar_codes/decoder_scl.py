"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _sc_decode_core, f_operation, g_operation, remap_channel_llr


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError('crc_length must be 8 or 16')


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    shift = crc_length - 1
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << shift
        for _ in range(crc_length):
            if reg & (1 << shift):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> i) & 1 for i in range(shift, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


def _path_metric_update(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    if bit != hard:
        pm += abs(llr)
    return pm


def _scl_recursive(llr_node, frozen_node, list_size):
    """递归 SCL，返回 [(pm, u_hat, u_up), ...]。"""

    n = len(llr_node)
    if n == 1:
        llr = float(llr_node[0])
        if frozen_node[0]:
            pm = 0.0 if llr >= 0 else abs(llr)
            u = np.array([0], dtype=np.int8)
            return [(pm, u, u.copy())]
        return [
            (_path_metric_update(0.0, llr, 0), np.array([0], dtype=np.int8), np.array([0], dtype=np.int8)),
            (_path_metric_update(0.0, llr, 1), np.array([1], dtype=np.int8), np.array([1], dtype=np.int8)),
        ]

    half = n // 2
    llr1 = llr_node[:half]
    llr2 = llr_node[half:]
    frozen1 = frozen_node[:half]
    frozen2 = frozen_node[half:]

    llr_upper = f_operation(llr1, llr2)
    upper_paths = _scl_recursive(llr_upper, frozen1, list_size)

    all_paths = []
    for pm_u, u1, u1_up in upper_paths:
        llr_lower = g_operation(llr1, llr2, u1_up)
        lower_paths = _scl_recursive(llr_lower, frozen2, list_size)
        for pm_l, u2, u2_up in lower_paths:
            u_hat = np.concatenate([u1, u2])
            u1_up_xor = (u1_up ^ u2_up).astype(np.int8)
            u_up = np.concatenate([u1_up_xor, u2_up])
            all_paths.append((pm_u + pm_l, u_hat, u_up))

    all_paths.sort(key=lambda item: item[0])
    return all_paths[:list_size]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = remap_channel_llr(llr_ch)

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = _sc_decode_core(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = _scl_recursive(llr_ch, self.frozen_bits, self.list_size)
        candidates = [(pm, u) for pm, u, _ in paths]

        if self.crc_length > 0:
            valid = [
                (pm, u) for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            pm, u_hat = min(valid if valid else candidates, key=lambda item: item[0])
        else:
            pm, u_hat = candidates[0]

        return u_hat.astype(int), float(pm)
