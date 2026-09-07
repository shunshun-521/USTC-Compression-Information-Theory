"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数"""
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 16):
            if crc_length <= 8:
                msb = (reg >> 7) & 1
                reg = ((reg << 1) & 0xFF) | (bit if _ == 0 else 0)
                if msb:
                    reg ^= poly
            else:
                msb = (reg >> 15) & 1
                reg = ((reg << 1) & 0xFFFF)
                if _ == 0:
                    reg |= int(bit)
                if msb:
                    reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            msb = (reg >> (crc_length - 1)) & 1
            reg = (reg << 1) & ((1 << crc_length) - 1)
            if msb:
                reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            msb = (reg >> (crc_length - 1)) & 1
            reg = (reg << 1) & ((1 << crc_length) - 1)
            if msb:
                reg ^= poly
    return reg == 0


class Path:
    """SCL 译码路径"""

    __slots__ = ("pm", "L", "C", "u_hat", "active")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _update_llr_layers(self, path, phi):
        if phi == 0:
            start_layer = 0
        else:
            start_layer = 0
            tmp = phi
            while tmp % 2 == 0:
                start_layer += 1
                tmp >>= 1

        for layer in range(self.n - 1, start_layer - 1, -1):
            stride = 1 << layer
            for block in range(0, self.N, 2 * stride):
                left = block
                right = block + stride
                for j in range(stride):
                    path.L[layer, left + j] = f_operation(
                        path.L[layer + 1, left + j], path.L[layer + 1, right + j]
                    )
                for j in range(stride):
                    u_partial = path.C[layer + 1, left + j]
                    path.L[layer, right + j] = g_operation(
                        path.L[layer + 1, left + j],
                        path.L[layer + 1, right + j],
                        u_partial,
                    )

    def _update_bit_layers(self, path, phi, u_bit):
        path.C[0, phi] = u_bit
        if phi % 2 == 1:
            layer = 0
            tmp = phi
            while tmp % 2 == 1:
                stride = 1 << layer
                for block in range(0, self.N, 2 * stride):
                    left = block
                    right = block + stride
                    for j in range(stride):
                        path.C[layer + 1, right + j] = (
                            path.C[layer, left + j] ^ path.C[layer, right + j]
                        )
                        path.C[layer + 1, left + j] = path.C[layer, right + j]
                layer += 1
                tmp >>= 1

    def _path_metric_penalty(self, llr, u_bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [Path(self.n, self.N) for _ in range(1)]
        paths[0].L[self.n] = llr_ch.copy()

        for phi in range(self.N):
            candidates = []

            for path in paths:
                self._update_llr_layers(path, phi)
                llr = path.L[0, phi]

                if self.frozen_bits[phi]:
                    penalty = self._path_metric_penalty(llr, 0)
                    path.pm += penalty
                    path.u_hat[phi] = 0
                    self._update_bit_layers(path, phi, 0)
                    candidates.append(path)
                else:
                    for u_bit in (0, 1):
                        new_path = Path(self.n, self.N)
                        new_path.pm = path.pm + self._path_metric_penalty(llr, u_bit)
                        new_path.L = path.L.copy()
                        new_path.C = path.C.copy()
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[phi] = u_bit
                        self._update_bit_layers(new_path, phi, u_bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_pm = None
        best_path = None

        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path
            if best_pm is None or path.pm < best_pm:
                best_pm = path.pm
                best_path = path

        chosen = best_crc if best_crc is not None else best_path
        return chosen.u_hat.copy(), chosen.pm
