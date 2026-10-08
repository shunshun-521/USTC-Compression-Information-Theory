"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, precompute_sc_indices, sc_decode

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _PathState:
    __slots__ = ("pm", "P", "C", "u_hat")

    def __init__(self, n, N, twoN, llr_ch):
        self.pm = 0.0
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.P[n, :] = llr_ch.copy()
        self.u_hat = []

    def copy(self):
        q = _PathState.__new__(_PathState)
        q.pm = self.pm
        q.C = self.C.copy()
        q.P = self.P.copy()
        q.u_hat = list(self.u_hat)
        return q


class SCLDecoder:
    """SCL 译码器（路径分裂时 Lazy Copy P/C）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = (
            precompute_sc_indices(N)
        )
        self.twoN = 2 * N

    def _llr_propagate_path(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            spm = 1 << layer
            left = (phi >> (layer + 1)) << (layer + 1)
            right = left + spm
            node = left >> layer
            path.P[layer, node] = f_operation(
                path.P[layer + 1, left], path.P[layer + 1, right]
            )
            path.P[layer, node + 1] = g_operation(
                path.P[layer + 1, left],
                path.P[layer + 1, right],
                path.C[layer, node],
            )

    def _bit_propagate(self, path, phi):
        for layer in self.bit_layer_vec[phi]:
            spm = 1 << layer
            left = (phi >> (layer + 1)) << (layer + 1)
            right = left + spm
            node = left >> layer
            path.C[layer + 1, right] = path.C[layer, node] ^ path.C[layer, node + 1]
            path.C[layer + 1, left] = path.C[layer, node + 1]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_PathState(self.n, self.N, self.twoN, llr_ch)]

        for phi in range(self.N):
            candidates = []
            for p in paths:
                self._llr_propagate_path(p, phi)
                llr_leaf = p.P[0, 0]
                frozen = self.frozen_bits[phi]

                if frozen:
                    child = p.copy()
                    u = 0
                    if llr_leaf < 0:
                        child.pm += abs(llr_leaf)
                    child.u_hat.append(u)
                    child.C[0, 0] = u
                    self._bit_propagate(child, phi)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = p.copy()
                        if u == 0 and llr_leaf < 0:
                            child.pm += abs(llr_leaf)
                        elif u == 1 and llr_leaf >= 0:
                            child.pm += abs(llr_leaf)
                        child.u_hat.append(u)
                        child.C[0, 0] = u
                        self._bit_propagate(child, phi)
                        candidates.append(child)

            candidates.sort(key=lambda x: x.pm)
            paths = candidates[: self.list_size]

        best = None
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(np.array(p.u_hat, dtype=int), self.crc_length)]
            if valid:
                best = min(valid, key=lambda x: x.pm)
        if best is None:
            best = min(paths, key=lambda x: x.pm)

        u_hat = np.array(best.u_hat, dtype=int)
        return u_hat, best.pm
