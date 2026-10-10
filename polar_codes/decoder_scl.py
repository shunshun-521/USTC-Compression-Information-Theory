"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _SCTree


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC-8 / CRC-16，返回信息比特 + CRC"""
    info_bits = np.asarray(info_bits, dtype=np.int8).reshape(-1)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).reshape(-1)
    if crc_length == 0:
        return True
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（树遍历 + 路径裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool).reshape(-1)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]
    def decode(self, llr_ch):
        tree = _SCTree(self.frozen, list_size=self.L_size)
        metrics, decisions = tree.decode(llr_ch)
        paths = list(zip(metrics, decisions))
        paths.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            for pm, u_hat in paths:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    return u_hat.copy(), pm
        return paths[0][1].copy(), paths[0][0]
