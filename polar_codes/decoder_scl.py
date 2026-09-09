"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode_recursive, sc_llr_at_phi

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_compute(info_bits, poly, crc_len):
    crc = 0
    for bit in info_bits:
        msb = (crc >> (crc_len - 1)) & 1
        if msb ^ int(bit):
            crc = ((crc << 1) ^ poly) & ((1 << crc_len) - 1)
        else:
            crc = (crc << 1) & ((1 << crc_len) - 1)
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc = _crc_compute(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_compute(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]
        self._llr = None

    def decode(self, llr_ch):
        if self.L == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        self._llr = np.asarray(llr_ch, dtype=np.float64)
        paths = [{'pm': 0.0, 'u': np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_phi = sc_llr_at_phi(self._llr, self.frozen_bits, path['u'], phi)
                if self.frozen_bits[phi]:
                    u_val = 0
                    penalty = abs(llr_phi) if llr_phi < 0 else 0.0
                    u_new = path['u'].copy()
                    u_new[phi] = u_val
                    new_paths.append({'pm': path['pm'] + penalty, 'u': u_new})
                else:
                    for u_val in (0, 1):
                        consistent = (u_val == 0 and llr_phi >= 0) or (
                            u_val == 1 and llr_phi < 0
                        )
                        penalty = 0.0 if consistent else abs(llr_phi)
                        u_new = path['u'].copy()
                        u_new[phi] = u_val
                        new_paths.append({
                            'pm': path['pm'] + penalty,
                            'u': u_new,
                        })

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.L]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u'][self.info_idx], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['u'], best['pm']
