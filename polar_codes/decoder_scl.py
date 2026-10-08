"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import polar_encode, bit_reversal_permutation
from decoder_sc import sc_decode, _llr_at_bit


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(8):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_len) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_len) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
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
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """
    SCL 译码器：在信息位上按可靠性分裂路径，路径度量为 BPSK-AWGN 对数似然。
    L=1 时退化为 SC（sc_decode）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _path_metric(self, llr_ch, u):
        x = polar_encode(u.astype(np.int8))
        sym = 1.0 - 2.0 * x
        return float(np.sum(np.log1p(np.exp(-llr_ch * sym))))

    def _sc_list_metric(self, llr_ch, u_prefix, phi, bit):
        y = llr_ch[self.rev]
        llr_phi = _llr_at_bit(y, u_prefix, phi)
        hard = 0 if llr_phi >= 0 else 1
        return 0.0 if hard == bit else abs(llr_phi)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        info_idx = np.where(~self.frozen_bits)[0]
        u0 = sc_decode(llr_ch, self.frozen_bits)
        paths = [(self._path_metric(llr_ch, u0), u0.astype(np.int8))]

        y = llr_ch[self.rev]
        reli_order = sorted(info_idx, key=lambda i: abs(y[i]))

        for phi in reli_order:
            new_paths = []
            for pm, u in paths:
                for bit in (0, 1):
                    u_new = u.copy()
                    u_new[phi] = bit
                    u_new[self.frozen_bits] = 0
                    pm_new = pm + self._sc_list_metric(llr_ch, u, phi, bit)
                    new_paths.append((pm_new, u_new))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda x: x[0])
        if self.crc_length > 0:
            for pm, u in paths:
                if crc_check(u, self.crc_length):
                    return u.astype(int), pm
        best_pm, best_u = paths[0]
        return best_u.astype(int), best_pm
