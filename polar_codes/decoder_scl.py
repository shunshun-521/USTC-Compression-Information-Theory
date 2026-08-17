"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode, _llr_at_bit


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if (reg >> (crc_length - 1)) & 1:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    received = 0
    for i, bit in enumerate(bits[-crc_length:]):
        received |= int(bit) << (crc_length - 1 - i)
    return remainder == received


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits)
        self.F = set(np.where(self.frozen_bits.astype(bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices) if info_indices is not None else None

    def _path_metric_update(self, pm, llr, u_val):
        hard = 1 if llr < 0 else 0
        if u_val != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size

        if L == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{'pm': 0.0, 'u': np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = _llr_at_bit(llr_ch, self.frozen_bits, path['u'], phi)

                if phi in self.F:
                    new_u = path['u'].copy()
                    new_u[phi] = 0
                    pm = self._path_metric_update(path['pm'], llr_phi, 0)
                    candidates.append({'pm': pm, 'u': new_u})
                else:
                    for u_val in (0, 1):
                        new_u = path['u'].copy()
                        new_u[phi] = u_val
                        pm = self._path_metric_update(path['pm'], llr_phi, u_val)
                        candidates.append({'pm': pm, 'u': new_u})

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:L]

        if self.crc_length > 0:
            def _crc_ok(path):
                payload = (path['u'][self.info_indices] if self.info_indices is not None
                           else path['u'])
                return crc_check(payload, self.crc_length)
            crc_ok = [p for p in paths if _crc_ok(p)]
            best = min(crc_ok if crc_ok else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u'].copy(), best['pm']
