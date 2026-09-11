"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive
from encoder import channel_llr_to_decoder


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divisor(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_divisor(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=int).ravel()
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（递归列表树搜索）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits.astype(int))
            return u_hat, 0.0

        llr = channel_llr_to_decoder(llr_ch, self.N)
        paths = self._decode_paths(llr, self.frozen_bits)

        best_crc = None
        best_all = min(paths, key=lambda p: p[0])

        if self.crc_length > 0:
            for pm, u_hat in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or pm < best_crc[0]:
                        best_crc = (pm, u_hat)

        chosen = best_crc if best_crc is not None else best_all
        return chosen[1].copy(), chosen[0]

    def _decode_paths(self, llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            idx = 0
            llr_val = float(llr_node[0])
            if frozen_node[0]:
                return [(0.0, np.array([0], dtype=int), np.array([0], dtype=int))]
            return [
                (_pm_penalty(llr_val, 0), np.array([0], dtype=int), np.array([0], dtype=int)),
                (_pm_penalty(llr_val, 1), np.array([1], dtype=int), np.array([1], dtype=int)),
            ]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        left_paths = self._decode_paths(llr_left, frozen_node[:half])

        combined = []
        for pm_left, u_left, u_left_up in left_paths:
            llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
            right_paths = self._decode_paths(llr_right, frozen_node[half:])
            for pm_right, u_right, u_right_up in right_paths:
                u_hat = np.concatenate([u_left, u_right])
                u_up_left = (u_left_up ^ u_right_up) & 1
                u_up = np.concatenate([u_up_left, u_right_up])
                combined.append((pm_left + pm_right, u_hat, u_up))

        combined.sort(key=lambda x: x[0])
        return combined[: self.list_size]
