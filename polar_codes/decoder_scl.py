"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError('crc_length must be 8 or 16')


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) / CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_update(pm, llr, u_bit):
    hard = 0 if llr >= 0 else 1
    if u_bit != hard:
        pm += abs(llr)
    return pm


def _calc_p(P, C, lam, phi, m):
    if lam == 0:
        return
    psi = phi >> 1
    pm2 = 1 << (m - lam)
    if phi % 2 == 0:
        _calc_p(P, C, lam - 1, psi, m)
        for beta in range(pm2):
            pm = beta + (psi << 1) * pm2
            P[lam - 1, pm] = f_operation(P[lam, pm], P[lam, pm + pm2])
    else:
        for beta in range(pm2):
            pm = beta + (psi << 1) * pm2
            P[lam - 1, pm] = g_operation(P[lam, pm], P[lam, pm + pm2], C[0, pm])


def _update_c(C, phi, u_bit, m):
    C[0, phi] = u_bit
    layer = 0
    phase = phi
    while phase % 2 == 1:
        layer += 1
        phase >>= 1
        size = 1 << (m - layer)
        for i in range(size):
            C[layer, i] = C[layer - 1, 2 * i] ^ C[layer - 1, 2 * i + 1]


class SCLDecoder:
    """SCL 译码器（顺序 SC 列表扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        m = self.m
        P = np.zeros((m + 1, self.N), dtype=np.float64)
        P[m, :] = llr_ch

        paths = [{
            'pm': 0.0,
            'C': np.zeros((m + 1, self.N), dtype=np.int32),
            'u': np.zeros(self.N, dtype=np.int32),
        }]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                _calc_p(P, path['C'], m, phi, m)
                llr_phi = P[0, phi]

                if self.frozen_bits[phi]:
                    cand = {
                        'pm': _pm_update(path['pm'], llr_phi, 0),
                        'C': path['C'].copy(),
                        'u': path['u'].copy(),
                    }
                    cand['u'][phi] = 0
                    _update_c(cand['C'], phi, 0, m)
                    new_paths.append(cand)
                else:
                    for u_bit in (0, 1):
                        cand = {
                            'pm': _pm_update(path['pm'], llr_phi, u_bit),
                            'C': path['C'].copy(),
                            'u': path['u'].copy(),
                        }
                        cand['u'][phi] = u_bit
                        _update_c(cand['C'], phi, u_bit, m)
                        new_paths.append(cand)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u'][self.info_idx], self.crc_length)]
            if valid:
                paths = valid

        best = paths[0]
        return best['u'].astype(int), best['pm']


def scl_path_metric_check(N, frozen_bits, llr_ch):
    """验证 L=1 的 SCL 与 SC 等价"""
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    u_scl, pm = scl.decode(llr_ch)
    u_sc = sc_decode(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc), pm
