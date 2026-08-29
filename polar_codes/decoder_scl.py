"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import _SCDCore, _bit_reversed


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（MSB-first）。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits[:-crc_length], poly, crc_length) == int(
        "".join(str(b) for b in bits[-crc_length:]), 2
    )


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时浅拷贝 LLR/比特树）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _llr_to_bit(llr):
        return 0 if llr >= 0 else 1

    @staticmethod
    def _pm_penalty(llr, u):
        return 0.0 if u == SCLDecoder._llr_to_bit(llr) else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N = self.N
        n = self.n
        decode_order = [_bit_reversed(i, n) for i in range(N)]

        paths = [{
            'pm': 0.0,
            'core': _SCDCore(N, self.frozen_bits),
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['core'].set_channel(llr_ch)

        for phi in decode_order:
            candidates = []
            for path in paths:
                core = path['core']
                core.update_llrs(phi)
                llr = core.current_llr(phi)

                if phi in self.frozen_set:
                    pm = path['pm'] + self._pm_penalty(llr, 0)
                    new_core = copy.copy(core)
                    new_core.L = core.L.copy()
                    new_core.B = core.B.copy()
                    new_core.B[phi, n] = 0
                    u_hat = path['u_hat'].copy()
                    u_hat[phi] = 0
                    new_core.update_bits(phi)
                    candidates.append({'pm': pm, 'core': new_core, 'u_hat': u_hat})
                else:
                    for u in (0, 1):
                        pm = path['pm'] + self._pm_penalty(llr, u)
                        new_core = copy.copy(core)
                        new_core.L = core.L.copy()
                        new_core.B = core.B.copy()
                        new_core.B[phi, n] = u
                        u_hat = path['u_hat'].copy()
                        u_hat[phi] = u
                        new_core.update_bits(phi)
                        candidates.append({'pm': pm, 'core': new_core, 'u_hat': u_hat})

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p['u_hat'][self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'], best['pm']
