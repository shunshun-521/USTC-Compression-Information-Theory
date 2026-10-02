"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _frozen_to_info_mask
import polar_lib_ref as plr

CRC_POLYS = {
    8: np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=np.int32),
    16: np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.int32),
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特末尾。"""
    return _crc_encode_py(np.asarray(info_bits, dtype=np.int8), crc_length)


def _crc_encode_py(info_bits, crc_length):
    poly = CRC_POLYS[crc_length]
    msg = np.concatenate([np.asarray(info_bits, dtype=np.int8), np.zeros(crc_length, dtype=np.int8)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i: i + crc_length] ^= poly[:crc_length]
    return np.concatenate([np.asarray(info_bits, dtype=np.int8), msg[-crc_length:]])


def crc_check(bits, crc_length=8):
    """校验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = _crc_encode_py(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（参考实现封装）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices, dtype=np.int64) if info_indices is not None else None
        if crc_length > 0 and self.info_indices is None:
            raise ValueError("CA-SCL 需要 info_indices")

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float32)
        if_info = _frozen_to_info_mask(self.frozen_bits).astype(np.int8)
        if self.crc_length > 0:
            k_crc = len(self.info_indices) - self.crc_length
            poly = CRC_POLYS[self.crc_length]
            u_hat = plr.SCLCRCDecoder(
                llr_ch,
                if_info,
                self.info_indices,
                k_crc,
                self.list_size,
                poly,
            )
        else:
            u_hat = plr.SCLDecoder(llr_ch, if_info, self.list_size)
        return u_hat.astype(int), 0.0
