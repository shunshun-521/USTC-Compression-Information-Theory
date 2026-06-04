"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, precompute_sc_indices, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        if reg & (1 << (degree - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
        else:
            reg = (reg << 1) & ((1 << degree) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly, degree = CRC8_POLY, 8
    elif crc_length == 16:
        poly, degree = CRC16_POLY, 16
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    remainder = _crc_remainder(info_bits, poly, degree)
    crc_bits = np.array(
        [(remainder >> (degree - 1 - i)) & 1 for i in range(degree)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly, degree = CRC8_POLY, 8
    elif crc_length == 16:
        poly, degree = CRC16_POLY, 16
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    return _crc_remainder(bits, poly, degree) == 0


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.rev = bit_reversal_permutation(N)
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]
        self._lambda_offset, self._llr_layers, self._bit_layers = precompute_sc_indices(N)

    def _new_path(self, llr_ch):
        P = [np.zeros(1 << l, dtype=np.float64) for l in range(self.n + 1)]
        C = [np.zeros(1 << l, dtype=int) for l in range(self.n + 1)]
        P[self.n][:] = llr_ch
        return {
            'P': P,
            'C': C,
            'PM': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
            'active': True,
        }

    def _copy_path(self, path):
        return {
            'P': [p.copy() for p in path['P']],
            'C': [c.copy() for c in path['C']],
            'PM': path['PM'],
            'u_hat': path['u_hat'].copy(),
            'active': True,
        }

    def _update_llr(self, path, phi):
        P, C = path['P'], path['C']
        n = self.n
        for layer in self._llr_layers[phi]:
            half = 1 << layer
            for block in range(half):
                left, right = block, block + half
                if (phi >> (n - 1 - layer)) & 1 == 0:
                    P[layer][block] = f_operation(P[layer + 1][left], P[layer + 1][right])
                else:
                    P[layer][block] = g_operation(
                        P[layer + 1][left],
                        P[layer + 1][right],
                        float(C[layer + 1][left]),
                    )

    def _propagate_bits(self, path, phi):
        C = path['C']
        for layer in self._bit_layers[phi]:
            half = 1 << layer
            for block in range(half):
                C[layer + 1][block + half] = C[layer + 1][block] ^ C[layer][block]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                self._update_llr(path, phi)
                llr0 = path['P'][0][0]

                if self.frozen_bits[phi]:
                    pen = _pm_penalty(llr0, 0)
                    path['PM'] += pen
                    path['u_hat'][phi] = 0
                    path['C'][0][0] = 0
                    self._propagate_bits(path, phi)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        pcopy = self._copy_path(path)
                        pen = _pm_penalty(llr0, u)
                        pcopy['PM'] += pen
                        pcopy['u_hat'][phi] = u
                        pcopy['C'][0][0] = u
                        self._propagate_bits(pcopy, phi)
                        new_paths.append(pcopy)

            new_paths.sort(key=lambda p: p['PM'])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p['PM'])

        return best['u_hat'].astype(int), best['PM']
