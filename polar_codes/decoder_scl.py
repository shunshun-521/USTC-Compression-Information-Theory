"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_update(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
    reg = ((reg << 1) | fb) & mask
    if fb:
        reg ^= poly
    return reg


def _crc_compute(bits, poly, crc_length):
    reg = 0
    for bit in np.asarray(bits, dtype=np.int8):
        reg = _crc_update(reg, bit, poly, crc_length)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_compute(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_compute(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def _scl_rec(self, llr, fzn, depth):
        if depth == 0:
            llr_val = float(llr[0])
            if fzn[0]:
                return [
                    {
                        "pm": self._pm_penalty(llr_val, 0),
                        "u": np.array([0], dtype=np.int8),
                        "u_up": np.array([0.0]),
                    }
                ]
            paths = []
            for bit in (0, 1):
                paths.append(
                    {
                        "pm": self._pm_penalty(llr_val, bit),
                        "u": np.array([bit], dtype=np.int8),
                        "u_up": np.array([float(bit)]),
                    }
                )
            return paths

        half = 1 << (depth - 1)
        llr_left = f_operation(llr[:half], llr[half:])
        upper_paths = self._scl_rec(llr_left, fzn[:half], depth - 1)

        all_paths = []
        for up in upper_paths:
            llr_right = g_operation(llr[:half], llr[half:], up["u_up"])
            lower_paths = self._scl_rec(llr_right, fzn[half:], depth - 1)
            for lp in lower_paths:
                u_left = up["u"]
                u_right = lp["u"]
                u_up = np.concatenate(
                    [
                        (
                            np.round(up["u_up"]).astype(np.int8)
                            ^ np.round(lp["u_up"]).astype(np.int8)
                        ).astype(np.float64),
                        lp["u_up"],
                    ]
                )
                all_paths.append(
                    {
                        "pm": up["pm"] + lp["pm"],
                        "u": np.concatenate([u_left, u_right]),
                        "u_up": u_up,
                    }
                )

        all_paths.sort(key=lambda p: p["pm"])
        return all_paths[: self.list_size]

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr, self.frozen_bits)
            return u_hat, 0.0

        paths = self._scl_rec(llr, self.frozen_bits, self.n)

        best = None
        if self.crc_length > 0:
            passed = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            if passed:
                best = min(passed, key=lambda p: p["pm"])
        if best is None:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"].copy(), best["pm"]
