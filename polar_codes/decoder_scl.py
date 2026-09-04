"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _update_llrs,
    _update_bits,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_divide(bits, poly, crc_length):
    """GF(2) 多项式除法求 CRC 余数"""
    msg = [int(b) for b in bits]
    gen = [1] + [((poly >> i) & 1) for i in range(crc_length - 1, -1, -1)]
    for i in range(len(msg) - crc_length):
        if msg[i] == 1:
            for j in range(len(gen)):
                msg[i + j] ^= gen[j]
    return msg[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_divide(padded, poly, crc_length)
    return np.concatenate([info_bits, np.array(remainder, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_divide(bits, poly, crc_length)
    return all(r == 0 for r in rem)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _metric_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _new_path(self, llr_ch):
        path = {
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
            "pm": 0.0,
        }
        path["L"][:, 0] = llr_ch
        return path

    def _copy_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
        }

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, self.n)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    pm = path["pm"] + self._metric_penalty(llr, bit)
                    candidates.append((pm, path, bit))
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._metric_penalty(llr, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, bit in candidates[: self.list_size]:
                child = self._copy_path(parent)
                child["pm"] = pm
                child["B"][l, self.n] = bit
                _update_bits(child["B"], l, self.n)
                new_paths.append(child)
            paths = new_paths

        decoded = []
        for path in paths:
            u_hat = path["B"][:, self.n].astype(int)
            decoded.append((path["pm"], u_hat))

        if self.crc_length > 0:
            valid = []
            for pm, u_hat in decoded:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        decoded.sort(key=lambda x: x[0])
        return decoded[0][1], decoded[0][0]
