"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _gf2_poly_divide(msg, poly, crc_len):
    """GF(2) 多项式除法，返回余数"""
    msg = np.array(msg, dtype=int).tolist()
    poly_bits = [(poly >> i) & 1 for i in range(crc_len, -1, -1)]
    data = msg + [0] * crc_len
    for i in range(len(msg)):
        if data[i] == 1:
            for j in range(len(poly_bits)):
                data[i + j] ^= poly_bits[j]
    return np.array(data[-crc_len:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _gf2_poly_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _gf2_poly_divide(bits.tolist(), poly, crc_length)
    return np.all(remainder == 0)


class _Path:
    """单条译码路径"""
    __slots__ = ('L', 'C', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch, br):
        self.L = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L[n] = llr_ch[br]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llr(self, path):
        for layer in range(self.n - 1, -1, -1):
            step = 1 << layer
            for i in range(0, self.N, 2 * step):
                for j in range(i, i + step):
                    path.L[layer, j] = f_operation(
                        path.L[layer + 1, j], path.L[layer + 1, j + step]
                    )
                    path.L[layer, j + step] = g_operation(
                        path.L[layer + 1, j],
                        path.L[layer + 1, j + step],
                        path.C[layer, j],
                    )

    def _update_bits(self, path, phi):
        for layer in range(self.n):
            step = 1 << layer
            for i in range(0, self.N, 2 * step):
                for j in range(i, i + step):
                    path.C[layer + 1, j] = path.C[layer, j] ^ path.C[layer, j + step]
                    path.C[layer + 1, j + step] = path.C[layer, j + step]

    def _path_metric_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch, self.br)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                self._update_llr(path)
                llr_val = path.L[0, phi]

                if self.frozen_bits[phi]:
                    new_path = self._copy_path(path)
                    new_path.pm += self._path_metric_penalty(llr_val, 0)
                    new_path.u_hat[phi] = 0
                    new_path.C[0, phi] = 0
                    self._update_bits(new_path, phi)
                    new_paths.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._path_metric_penalty(llr_val, u_val)
                        new_path.u_hat[phi] = u_val
                        new_path.C[0, phi] = u_val
                        self._update_bits(new_path, phi)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat[self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm

    def _copy_path(self, path):
        new = _Path.__new__(_Path)
        new.L = path.L.copy()
        new.C = path.C.copy()
        new.pm = path.pm
        new.u_hat = path.u_hat.copy()
        return new
