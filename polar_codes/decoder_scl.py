"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation, _b_check, _s_updater, _compute_llr, _UNINIT,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length):
    if crc_length == 8:
        poly, width = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in bits:
        msb = (reg >> (width - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << width) - 1)
        if msb ^ int(bit):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ('pm', 'llrs', 's', 'u_hat')

    def __init__(self, N, n):
        self.pm = 0.0
        self.llrs = np.full((n + 1, N), _UNINIT, dtype=np.float64)
        self.s = -np.ones((n + 1, N), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = _Path(self.llrs.shape[1], self.llrs.shape[0] - 1)
        p.pm = self.pm
        p.llrs[:] = self.llrs
        p.s[:] = self.s
        p.u_hat[:] = self.u_hat
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 llrs/s）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_add(pm, llr, u):
        if (u == 0 and llr >= 0) or (u == 1 and llr < 0):
            return pm
        return pm + abs(llr)

    def _crc_pass(self, u_hat):
        if self.crc_length == 0:
            return True
        info = u_hat[self.info_indices]
        return crc_check(info, self.crc_length)

    def decode(self, llr_ch):
        N, n = self.N, self.n
        br = bit_reversal_permutation(N)
        llr = llr_ch[br]

        paths = [_Path(N, n)]
        paths[0].llrs[n, :] = llr

        for i in range(N):
            extensions = []
            for path in paths:
                llr_i = _compute_llr(0, i, path.llrs, path.s)

                if self.frozen_bits[i]:
                    new_pm = self._pm_add(path.pm, llr_i, 0)
                    path.pm = new_pm
                    path.u_hat[i] = 0
                    path.s[0, i] = 0
                    extensions.append((new_pm, path, None))
                else:
                    for u in (0, 1):
                        new_pm = self._pm_add(path.pm, llr_i, u)
                        extensions.append((new_pm, path, u))

            extensions.sort(key=lambda x: x[0])
            new_paths = []
            for new_pm, parent, u_val in extensions:
                if len(new_paths) >= self.list_size:
                    break
                if u_val is None:
                    p = parent
                else:
                    p = parent.copy()
                    p.pm = new_pm
                    p.u_hat[i] = u_val
                    p.s[0, i] = u_val
                new_paths.append(p)

            paths = new_paths

        crc_paths = [p for p in paths if self._crc_pass(p.u_hat)]
        if crc_paths:
            best = min(crc_paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
