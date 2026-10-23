"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _sc_decode_core, f_operation, g_operation, precompute_sc_indices


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
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("P", "C", "pm", "u_hat", "active")

    def __init__(self, n, N):
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径复制在分裂时执行）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        _, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _update_llr(self, path, phi):
        layer_start = 0
        tmp = phi
        while tmp & 1:
            tmp >>= 1
            layer_start += 1

        for layer in range(layer_start, self.n):
            step = 1 << (self.n - layer)
            half = step // 2
            block = (phi // step) * step
            for j in range(half):
                u = block + j
                path.P[layer, u] = f_operation(
                    path.P[layer + 1, u], path.P[layer + 1, u + half]
                )
        for layer in range(self.n - 1, layer_start - 1, -1):
            step = 1 << (self.n - layer)
            half = step // 2
            block = (phi // step) * step
            for j in range(half):
                u = block + j
                path.P[layer, u + half] = g_operation(
                    path.P[layer + 1, u],
                    path.P[layer + 1, u + half],
                    path.C[layer, u],
                )

    def _update_bits(self, path, phi):
        path.C[self.n, phi] = path.u_hat[phi]
        for layer in self.bit_layer_vec[phi]:
            idx = ((phi >> (layer + 1)) << (layer + 1)) + ((1 << layer) - 1)
            path.C[layer, idx] = path.C[layer + 1, idx]

    def _pm_penalty(self, llr, u):
        v = 0 if llr >= 0 else 1
        return 0.0 if u == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]
        if self.list_size == 1 and self.crc_length == 0:
            return _sc_decode_core(llr_ch, self.frozen_bits), 0.0
        paths = [_Path(self.n, self.N)]
        paths[0].P[self.n, :] = llr_ch

        for phi in range(self.N):
            candidates = []
            for pidx, path in enumerate(paths):
                if not path.active:
                    continue
                self._update_llr(path, phi)
                llr_bit = path.P[0, 0]

                if self.frozen_bits[phi]:
                    pen = self._pm_penalty(llr_bit, 0)
                    path.pm += pen
                    path.u_hat[phi] = 0
                    self._update_bits(path, phi)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        new_path = _Path(self.n, self.N)
                        new_path.P = path.P.copy()
                        new_path.C = path.C.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.pm = path.pm + self._pm_penalty(llr_bit, u)
                        new_path.u_hat[phi] = u
                        self._update_bits(new_path, phi)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best = paths[0]
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or p.pm < best_crc.pm:
                        best_crc = p
        if best_crc is not None:
            return best_crc.u_hat.copy(), best_crc.pm
        return best.u_hat.copy(), best.pm
