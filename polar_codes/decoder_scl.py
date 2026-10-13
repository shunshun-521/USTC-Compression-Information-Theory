"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

import ref_crc as _ref_crc
import ref_function as _ref_function

_ref_path = Path(__file__).resolve().parent / "polar_sc_ref.py"
_spec = importlib.util.spec_from_file_location("polar_sc_ref", _ref_path)
_polar_ref = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("function", _ref_function)
sys.modules.setdefault("CRC", _ref_crc)
_spec.loader.exec_module(_polar_ref)

from decoder_sc import _frozen_to_info_pos, _llr_for_decoder


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07  # x^8 + x^2 + x + 1
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（MSB 先传）"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length=crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（列表路径由参考实现维护）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_pos = _frozen_to_info_pos(self.frozen_bits)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self._frozen_value = 0

    def decode(self, llr_ch):
        llr = _llr_for_decoder(llr_ch)
        decode_para = [self.list_size, "hf"]
        u_d = _polar_ref.scl_decoder(
            llr,
            self.info_pos,
            self._frozen_value,
            decode_para,
            self.crc_length,
        )
        u_hat = np.asarray(u_d, dtype=np.int8)
        return u_hat, 0.0
