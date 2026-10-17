"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _SCLCore, channel_llr_to_decoder


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_positions = np.flatnonzero(~self.frozen_bits)

    def decode(self, llr_ch):
        llr = channel_llr_to_decoder(llr_ch)
        core = _SCLCore(self.frozen_bits, list_size=self.L)
        core.metrics = [0.0]
        core.decisions = [np.zeros(self.N, dtype=np.int8)]
        core._node([llr], 0, self.N)

        paths = list(zip(core.metrics, core.decisions, strict=True))
        paths.sort(key=lambda x: x[0])

        if self.crc_length == 0:
            pm, u_hat = paths[0]
            return u_hat.astype(int), pm

        best_crc = None
        for pm, u_hat in paths:
            payload = u_hat[self.info_positions]
            if crc_check(payload, self.crc_length):
                if best_crc is None or pm < best_crc[1]:
                    best_crc = (u_hat.copy(), pm)

        if best_crc is not None:
            return best_crc[0].astype(int), best_crc[1]
        pm, u_hat = paths[0]
        return u_hat.astype(int), pm
