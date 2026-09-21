"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _depermute_llr, f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _scl_decode_paths(llr, frozen_bits, list_size):
    """递归 SCL 译码，返回 (pm, u_hat, u_up) 路径列表。"""
    n = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    if n == 1:
        if frozen_bits[0]:
            u = np.array([0], dtype=int)
            return [(0.0, u, u)]
        llr_val = llr[0]
        hard = 0 if llr_val >= 0 else 1
        paths = []
        for u_bit in (0, 1):
            pm = 0.0 if u_bit == hard else abs(llr_val)
            u = np.array([u_bit], dtype=int)
            paths.append((pm, u, u))
        paths.sort(key=lambda item: item[0])
        return paths[:list_size]

    half = n // 2
    llr_left = f_operation(llr[:half], llr[half:])
    left_paths = _scl_decode_paths(llr_left, frozen_bits[:half], list_size)

    all_paths = []
    for pm_left, u_left, u_left_up in left_paths:
        llr_right = g_operation(llr[:half], llr[half:], u_left_up)
        right_paths = _scl_decode_paths(llr_right, frozen_bits[half:], list_size)
        for pm_right, u_right, u_right_up in right_paths:
            u_hat = np.concatenate([u_left, u_right])
            u_up = np.concatenate([
                (u_left_up.astype(int) ^ u_right_up.astype(int)).astype(int),
                u_right_up.astype(int),
            ])
            all_paths.append((pm_left + pm_right, u_hat, u_up))

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
        """主译码函数。"""
        llr = _depermute_llr(np.asarray(llr_ch, dtype=np.float64))

        paths = [
            (pm, u) for pm, u, _ in _scl_decode_paths(
                llr, self.frozen_bits, self.list_size
            )
        ]

        if self.crc_length > 0:
            valid = [
                (pm, u) for pm, u in paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                pm, u_hat = min(valid, key=lambda item: item[0])
            else:
                pm, u_hat = paths[0]
        else:
            pm, u_hat = paths[0]

        return np.asarray(u_hat, dtype=int).copy(), pm
