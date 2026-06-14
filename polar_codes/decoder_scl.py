"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, precompute_sc_indices, sc_decode


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB first）"""
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================

class Path:
    __slots__ = ("pm", "P", "C", "u_hat")

    def __init__(self, n, N, llr_ch=None):
        self.pm = 0.0
        self.P = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
        self.C = [np.zeros(N, dtype=np.int32) for _ in range(n + 1)]
        self.u_hat = np.zeros(N, dtype=int)
        if llr_ch is not None:
            self.P[n][:] = llr_ch


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = (
            precompute_sc_indices(N)
        )
        br = bit_reversal_permutation(N)
        self.inv_br = np.empty(N, dtype=int)
        for i, j in enumerate(br):
            self.inv_br[j] = i

    def _update_llrs(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            offset = self.lambda_offset[layer]
            for i in range(0, self.N, 2 * offset):
                for j in range(offset):
                    left = i + j
                    right = left + offset
                    if phi & offset:
                        path.P[layer][left] = g_operation(
                            path.P[layer + 1][left],
                            path.P[layer + 1][right],
                            path.C[layer][left],
                        )
                    else:
                        path.P[layer][left] = f_operation(
                            path.P[layer + 1][left], path.P[layer + 1][right]
                        )
                        path.P[layer][right] = path.P[layer + 1][right]

    def _propagate_bits(self, path, phi):
        for layer in self.bit_layer_vec[phi]:
            offset = self.lambda_offset[layer]
            for i in range(0, self.N, 2 * offset):
                for j in range(offset):
                    left = i + j
                    right = left + offset
                    path.C[layer + 1][right] = path.C[layer][left] ^ path.C[layer][right]
                    path.C[layer + 1][left] = path.C[layer][left]

    def _llr_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.inv_br]
        paths = [Path(self.n, self.N, llr_ch)]

        for phi in range(self.N):
            is_frozen = self.frozen_bits[phi]
            candidates = []

            for path in paths:
                self._update_llrs(path, phi)
                llr = path.P[0][0]

                if is_frozen:
                    new_path = copy.deepcopy(path)
                    new_path.pm += self._llr_penalty(llr, 0)
                    new_path.u_hat[phi] = 0
                    new_path.C[0][0] = 0
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = copy.deepcopy(path)
                        new_path.pm += self._llr_penalty(llr, u)
                        new_path.u_hat[phi] = u
                        new_path.C[0][0] = u
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            for path in paths:
                self._propagate_bits(path, phi)

        best = paths[0]
        if self.crc_length > 0:
            crc_pass = [p for p in paths if self._crc_pass(p)]
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm

    def _crc_pass(self, path):
        info_positions = np.where(~self.frozen_bits)[0]
        return crc_check(path.u_hat[info_positions], self.crc_length)
