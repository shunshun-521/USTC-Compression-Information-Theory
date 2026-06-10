"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _get_sc_tables, f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    """对 bits 做 CRC 除法，返回 crc_length 位余数。"""
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & msb:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_process(msg, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("pm", "P", "C", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.P = np.zeros((n + 1, 2 * N), dtype=np.float64)
        self.C = np.zeros((n + 1, 2 * N), dtype=np.int32)
        self.P[n, :N] = llr_ch
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 P/C）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_mask = ~self.frozen_bits
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = _get_sc_tables(N)

    def _branch_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _copy_path(self, path):
        new_path = _Path(self.N, self.n, np.zeros(self.N))
        new_path.pm = path.pm
        new_path.P = path.P.copy()
        new_path.C = path.C.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_llr(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            psi = phi >> layer
            delta = self.lambda_offset[layer]
            for omega in range(delta):
                idx = omega + psi * 2 * delta
                path.P[layer, idx] = f_operation(
                    path.P[layer + 1, idx], path.P[layer + 1, idx + delta]
                )
                path.P[layer, idx + delta] = g_operation(
                    path.P[layer + 1, idx],
                    path.P[layer + 1, idx + delta],
                    path.C[layer, idx],
                )

    def _update_bits(self, path, phi):
        for layer in self.bit_layer_vec[phi]:
            psi = phi >> layer
            delta = self.lambda_offset[layer]
            omega = psi % 2
            idx = (psi // 2) * 2 * delta + omega * delta
            path.C[layer + 1, 2 * idx] = path.C[layer, idx]
            path.C[layer + 1, 2 * idx + delta] = (
                path.C[layer, idx] ^ path.C[layer, idx + delta]
            )

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._update_llr(path, phi)
                llr = path.P[0, phi * 2]

                if self.frozen_bits[phi]:
                    path.pm += self._branch_penalty(llr, 0)
                    path.u_hat[phi] = 0
                    path.C[0, phi * 2] = 0
                    self._update_bits(path, phi)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._branch_penalty(llr, bit)
                        new_path.u_hat[phi] = bit
                        new_path.C[0, phi * 2] = bit
                        self._update_bits(new_path, phi)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_paths = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_mask], self.crc_length)
            ]
            pool = crc_paths if crc_paths else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
