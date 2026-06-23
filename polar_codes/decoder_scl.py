"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, sc_decode
from encoder import bit_reversal_permutation


_CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_divisor(crc_length):
    poly = _CRC_POLYS[crc_length]
    return np.array([int(b) for b in f"1{poly:0{crc_length}b}"], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    divisor = _crc_divisor(crc_length)
    r = crc_length
    msg = np.concatenate([info_bits, np.zeros(r, dtype=np.int8)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i : i + r + 1] ^= divisor
    return np.concatenate([info_bits, msg[-r:]])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    divisor = _crc_divisor(crc_length)
    r = crc_length
    msg = bits.copy()
    for i in range(len(bits) - r):
        if msg[i] == 1:
            msg[i : i + r + 1] ^= divisor
    return np.all(msg[-r:] == 0)


def _left_sweep(L, R, n, N):
    for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for block in range(0, N, 2 * s):
            for i in range(block, block + s):
                i2 = i + s
                L[i, j - 1] = f_operation(R[i, j] + L[i2, j], L[i, j])
                L[i2, j - 1] = f_operation(R[i, j], L[i, j]) + L[i2, j]


class _Path:
    __slots__ = ("pm", "u_hat", "R0")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.R0 = np.zeros(N, dtype=np.float64)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时仅复制 u_hat 与 R0）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_idx = np.where(self.frozen_bits == 1)[0]
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.LARGE = 1e6
        self._L = np.zeros((N, self.n + 1), dtype=np.float64)
        self._R = np.zeros((N, self.n + 1), dtype=np.float64)

    def _current_llr(self, llr_ch, path, phi):
        N, n = self.N, self.n
        L, R = self._L, self._R
        L[:, n] = llr_ch[self.br]
        R.fill(0.0)
        R[:, 0] = path.R0
        R[self.frozen_idx, 0] = self.LARGE
        _left_sweep(L, R, n, N)
        return L[phi, 0] + R[phi, 0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N)]
        paths[0].R0[self.frozen_idx] = self.LARGE

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr0 = self._current_llr(llr_ch, path, phi)

                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr0 >= 0 else abs(llr0)
                    new_path = _Path(self.N)
                    new_path.u_hat = path.u_hat.copy()
                    new_path.R0 = path.R0.copy()
                    new_path.pm = path.pm + penalty
                    new_path.u_hat[phi] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr0 >= 0) or (bit == 1 and llr0 < 0) else abs(llr0)
                        new_path = _Path(self.N)
                        new_path.u_hat = path.u_hat.copy()
                        new_path.R0 = path.R0.copy()
                        new_path.pm = path.pm + penalty
                        new_path.u_hat[phi] = bit
                        if bit == 0:
                            new_path.R0[phi] = self.LARGE
                        else:
                            new_path.R0[phi] = -self.LARGE
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_idx = 0
        best_pm = paths[0].pm
        crc_pass = []

        for i, path in enumerate(paths):
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append((path.pm, i))
            if path.pm < best_pm:
                best_pm = path.pm
                best_idx = i

        if self.crc_length > 0 and crc_pass:
            crc_pass.sort(key=lambda x: x[0])
            best_idx = crc_pass[0][1]
            best_pm = crc_pass[0][0]

        return paths[best_idx].u_hat, best_pm
