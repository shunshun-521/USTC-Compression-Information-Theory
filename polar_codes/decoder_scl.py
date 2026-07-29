"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    _update_llr,
    _update_bits,
)


CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]


def _get_poly_bits(crc_length):
    if crc_length == 8:
        return CRC8_POLY_BITS
    if crc_length == 16:
        # CRC-16-IBM: x^16 + x^15 + x^2 + 1 (0x8005)
        bits = [0] * 17
        bits[0] = bits[1] = bits[15] = bits[16] = 1
        return bits
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_remainder_gf2(bits, crc_length, poly_bits):
    msg = list(np.asarray(bits, dtype=int)) + [0] * crc_length
    n = len(bits)
    for i in range(n):
        if msg[i]:
            for j, pb in enumerate(poly_bits):
                if pb and i + j < len(msg):
                    msg[i + j] ^= 1
    return np.array(msg[n:n + crc_length], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _get_poly_bits(crc_length)
    crc_bits = _crc_remainder_gf2(info_bits, crc_length, poly)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = _get_poly_bits(crc_length)
    msg = list(bits)
    n = len(bits) - crc_length
    for i in range(n):
        if msg[i]:
            for j, pb in enumerate(poly):
                if pb and i + j < len(msg):
                    msg[i + j] ^= 1
    return all(x == 0 for x in msg[-crc_length:])


class PathState:
    """单条译码路径（Lazy Copy）。"""

    def __init__(self, N, n, pm=0.0):
        self.N = N
        self.n = n
        self.pm = pm
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.u_hat = np.zeros(N, dtype=int)
        self.L_owned = True
        self.B_owned = True

    def copy(self):
        new = PathState(self.N, self.n, self.pm)
        new.L = self.L
        new.B = self.B
        new.u_hat = self.u_hat.copy()
        new.L_owned = False
        new.B_owned = False
        return new

    def ensure_L(self):
        if not self.L_owned:
            self.L = self.L.copy()
            self.L_owned = True

    def ensure_B(self):
        if not self.B_owned:
            self.B = self.B.copy()
            self.B_owned = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.inv_brp = np.argsort(bit_reversal_permutation(N))

    @staticmethod
    def _pm_penalty(llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ordered = llr_ch[self.inv_brp]

        path = PathState(self.N, self.n, pm=0.0)
        path.L[:, 0] = llr_ordered
        paths = [path]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for p in paths:
                p.ensure_L()
                p.ensure_B()
                _update_llr(p.L, p.B, l, self.n, self.N)
                llr_val = p.L[l, self.n]

                if self.frozen_bits[l]:
                    np_ = p.copy()
                    np_.pm += self._pm_penalty(llr_val, 0)
                    np_.ensure_B()
                    np_.B[l, self.n] = 0
                    np_.u_hat[l] = 0
                    _update_bits(np_.B, l, self.n, self.N)
                    candidates.append(np_)
                else:
                    for u_val in (0, 1):
                        np_ = p.copy()
                        np_.pm += self._pm_penalty(llr_val, u_val)
                        np_.ensure_B()
                        np_.B[l, self.n] = u_val
                        np_.u_hat[l] = u_val
                        _update_bits(np_.B, l, self.n, self.N)
                        candidates.append(np_)

            candidates.sort(key=lambda x: x.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda x: x.pm)
        else:
            best = min(paths, key=lambda x: x.pm)

        return best.u_hat.copy(), best.pm
