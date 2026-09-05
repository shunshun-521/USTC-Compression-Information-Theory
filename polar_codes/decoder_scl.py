"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import compute_sc_llr_at_phase, sc_decode


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_run(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_run(info_bits, crc_length)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_run(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u_val):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_val == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{'u_hat': np.zeros(self.N, dtype=int), 'pm': 0.0}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = compute_sc_llr_at_phase(llr_ch, path['u_hat'], phi, self.frozen_bits)

                if self.frozen_bits[phi]:
                    new_path = {'u_hat': path['u_hat'].copy(), 'pm': path['pm'] + self._pm_penalty(llr, 0)}
                    new_path['u_hat'][phi] = 0
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = {'u_hat': path['u_hat'].copy(), 'pm': path['pm'] + self._pm_penalty(llr, u_val)}
                        new_path['u_hat'][phi] = u_val
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'][self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
