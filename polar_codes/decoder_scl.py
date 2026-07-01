"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import _active_bit_level, _active_llr_level, _bit_reversed, _update_bits, _update_llrs
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class Path:
    """单条译码路径。"""

    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        br = bit_reversal_permutation(N)
        self.L[:, 0] = llr_ch[br]

    def clone(self):
        new = Path.__new__(Path)
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        new.L = self.L.copy()
        new.B = self.B.copy()
        return new


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices, dtype=int) if info_indices is not None else None
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _crc_pass(self, u_hat):
        if self.crc_length == 0:
            return True
        payload = u_hat[self.info_indices]
        return crc_check(payload, self.crc_length)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                _update_llrs(l, path.L, path.B, self.n, self.N)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr_val, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(l, path.B, self.n, self.N)
                    candidates.append(path)
                else:
                    for u_bit in (0, 1):
                        new_path = path.clone()
                        new_path.pm += self._pm_penalty(llr_val, u_bit)
                        new_path.u_hat[l] = u_bit
                        new_path.B[l, self.n] = u_bit
                        _update_bits(l, new_path.B, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [p for p in paths if self._crc_pass(p.u_hat)]
            best = min(crc_pass if crc_pass else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
