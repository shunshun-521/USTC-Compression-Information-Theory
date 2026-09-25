"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
    _update_bits_pdf,
    _update_llrs_pdf,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _pm_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -1e2, 1e2)
        paths = [{
            'L': np.full((self.N, self.n + 1), np.nan),
            'B': np.full((self.N, self.n + 1), np.nan),
            'pm': 0.0,
        }]
        paths[0]['L'][:, self.n] = llr_ch

        for phi in range(self.N):
            x = bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs_pdf(path['L'], path['B'], x, self.n)
                llr_val = path['L'][x, 0]
                if np.isnan(llr_val):
                    llr_val = 0.0

                if x in self.frozen_set:
                    npath = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._pm_penalty(llr_val, 0),
                    }
                    npath['B'][x, 0] = 0
                    _update_bits_pdf(npath['B'], x, self.n)
                    new_paths.append(npath)
                else:
                    for u_val in (0, 1):
                        npath = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._pm_penalty(llr_val, u_val),
                        }
                        npath['B'][x, 0] = u_val
                        _update_bits_pdf(npath['B'], x, self.n)
                        new_paths.append(npath)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        crc_pass = []
        for path in paths:
            u_hat = np.nan_to_num(path['B'][:, 0], nan=0).astype(int)
            info_bits = u_hat[self.info_indices]
            if self.crc_length > 0:
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            else:
                crc_pass.append(path)

        pool = crc_pass if crc_pass else paths
        best = min(pool, key=lambda p: p['pm'])
        return np.nan_to_num(best['B'][:, 0], nan=0).astype(int).copy(), best['pm']
