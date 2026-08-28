"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _slow_llr


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYS[crc_length]
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in np.asarray(info_bits, dtype=int):
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    poly = CRC_POLYS[crc_length]
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in np.asarray(bits, dtype=int):
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, u_val):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_val == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        paths = [{'u_hat': np.zeros(N, dtype=int), 'pm': 0.0}]

        for i in range(N):
            new_paths = []
            for path in paths:
                llr = _slow_llr(i, N, llr_ch, path['u_hat'][:i])
                if self.frozen_bits[i]:
                    p = {'u_hat': path['u_hat'].copy(), 'pm': path['pm'] + self._pm_penalty(llr, 0)}
                    p['u_hat'][i] = 0
                    new_paths.append(p)
                else:
                    for u_val in (0, 1):
                        p = {'u_hat': path['u_hat'].copy(), 'pm': path['pm'] + self._pm_penalty(llr, u_val)}
                        p['u_hat'][i] = u_val
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])
        return best['u_hat'].copy(), best['pm']
