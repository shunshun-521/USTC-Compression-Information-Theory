"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    sc_decode_recursive,
    _node_pair,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
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
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    __slots__ = ("pm", "P", "C", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        _, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _copy_path(self, path):
        new_path = _Path(self.N, self.n)
        new_path.pm = path.pm
        new_path.P = path.P.copy()
        new_path.C = path.C.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llr(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            left, right = _node_pair(phi, layer)
            if (phi >> layer) & 1 == 0:
                path.P[layer, left] = f_operation(
                    path.P[layer + 1, left], path.P[layer + 1, right]
                )
            else:
                path.P[layer, right] = g_operation(
                    path.P[layer + 1, left],
                    path.P[layer + 1, right],
                    path.C[layer, left],
                )

    def _update_bits(self, path, phi):
        for layer in self.bit_layer_vec[phi]:
            left, right = _node_pair(phi, layer)
            path.C[layer + 1, right] = path.C[layer, right]
            path.C[layer + 1, left] = path.C[layer, left] ^ path.C[layer, right]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits.astype(bool))
            return u_hat, 0.0

        paths = [_Path(self.N, self.n)]
        paths[0].P[self.n, : self.N] = llr_ch

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._update_llr(path, phi)
                llr = path.P[0, phi]

                if self.frozen_bits[phi]:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.u_hat[phi] = 0
                    new_path.C[0, phi] = 0
                    self._update_bits(new_path, phi)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.u_hat[phi] = bit
                        new_path.C[0, phi] = bit
                        self._update_bits(new_path, phi)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
