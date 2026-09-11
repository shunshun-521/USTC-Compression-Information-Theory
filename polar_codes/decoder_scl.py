"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0.0 else 1
    return 0.0 if bit == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（递归列表实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _scl_rec(self, llr_node, frozen_node):
        n_len = len(llr_node)
        if n_len == 1:
            if frozen_node[0]:
                bit = np.array([0], dtype=np.int8)
                return [(bit, bit.astype(float), 0.0)]
            return [
                (np.array([0], dtype=np.int8), np.array([0.0]), _path_metric_penalty(llr_node[0], 0)),
                (np.array([1], dtype=np.int8), np.array([1.0]), _path_metric_penalty(llr_node[0], 1)),
            ]

        half = n_len // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        upper_paths = self._scl_rec(llr_left, frozen_node[:half])
        all_paths = []

        for u_left, u_left_up, pm_left in upper_paths:
            llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
            lower_paths = self._scl_rec(llr_right, frozen_node[half:])
            for u_right, u_right_up, pm_right in lower_paths:
                u_hat = np.concatenate([u_left, u_right])
                u_left_up_int = np.mod(u_left_up.astype(int) + u_right_up.astype(int), 2)
                u_up = np.concatenate([u_left_up_int.astype(float), u_right_up])
                all_paths.append((u_hat, u_up, pm_left + pm_right))

        all_paths.sort(key=lambda x: x[2])
        return [(u, up, pm) for u, up, pm in all_paths[: self.list_size]]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._scl_rec(llr_ch, self.frozen_bits)

        if self.crc_length > 0:
            valid = []
            for u_hat, _, pm in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((u_hat, pm))
            if valid:
                best = min(valid, key=lambda x: x[1])
            else:
                best = (paths[0][0], paths[0][2])
        else:
            best = (paths[0][0], paths[0][2])

        return best[0].copy(), best[1]
