"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import _update_llrs, _update_bits


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    data = bits[:-crc_length]

    reg = 0
    for bit in data:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    computed = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.array_equal(computed, bits[-crc_length:])


class Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, n, N):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [Path(n, N)]
        paths[0].L[:, 0] = llr_ch.copy()

        for i in range(N):
            l = self.rev[i]
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                llr = path.L[l, n]

                if self.frozen_bits[l]:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.B[l, n] = 0
                    new_path.u_hat[l] = 0
                    _update_bits(new_path.B, l, n, N)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, u)
                        new_path.B[l, n] = u
                        new_path.u_hat[l] = u
                        _update_bits(new_path.B, l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        best_path = self._select_best_path(paths)
        return best_path.u_hat.copy(), best_path.pm

    def _copy_path(self, path):
        new_path = Path(self.n, self.N)
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                return min(crc_pass, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
