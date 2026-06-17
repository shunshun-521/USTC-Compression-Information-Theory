"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, precompute_sc_indices, sc_decode_recursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_mod2_divide(message, poly, crc_length):
    """对 message||0^r 做模 2 长除法，返回 CRC 余数位。"""
    msg = np.concatenate([np.asarray(message, dtype=int), np.zeros(crc_length, dtype=int)])
    poly_bits = np.array([(poly >> (crc_length - i)) & 1 for i in range(crc_length + 1)], dtype=int)
    for i in range(len(message)):
        if msg[i] == 1:
            msg[i:i + crc_length + 1] ^= poly_bits
    return msg[-crc_length:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_mod2_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_mod2_divide(bits, poly, crc_length)
    return np.all(rem == 0)


class _Path:
    __slots__ = ("pm", "u_hat", "L", "C")

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int32)


class SCLDecoder:
    """SCL 译码器（路径复制 + L/C 分层存储）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _copy_path(self, path):
        new_path = _Path(self.n, self.N)
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        new_path.L = path.L.copy()
        new_path.C = path.C.copy()
        return new_path

    def _update_path(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            step = self.lambda_offset[layer]
            for block in range(0, self.N, 2 * step):
                for j in range(step):
                    i = block + j
                    path.L[layer, i] = f_operation(
                        path.L[layer + 1, i], path.L[layer + 1, i + step]
                    )

        g_layer = len(self.llr_layer_vec[phi])
        if g_layer <= self.n - 1:
            step = self.lambda_offset[g_layer]
            block = (phi // (2 * step)) * (2 * step)
            offset = phi % step
            i = block + offset
            path.L[g_layer, i + step] = g_operation(
                path.L[g_layer + 1, i],
                path.L[g_layer + 1, i + step],
                path.C[g_layer, i],
            )

        return path.L[0, 0]

    def _propagate_bits(self, path, phi):
        path.C[0, 0] = path.u_hat[phi]
        for layer in self.bit_layer_vec[phi]:
            step = self.lambda_offset[layer]
            for block in range(0, self.N, 2 * step):
                for j in range(step):
                    i = block + j
                    path.C[layer + 1, i] = path.C[layer, i] ^ path.C[layer, i + step]
                    path.C[layer + 1, i + step] = path.C[layer, i + step]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.n, self.N)]
        paths[0].L[self.n, :] = llr_ch

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr = self._update_path(path, phi)
                if self.frozen_bits[phi]:
                    branch_bits = [0]
                else:
                    branch_bits = [0, 1]

                for u in branch_bits:
                    hard = 0 if llr >= 0 else 1
                    pm = path.pm + (0.0 if u == hard else abs(llr))
                    new_path = self._copy_path(path)
                    new_path.pm = pm
                    new_path.u_hat[phi] = u
                    self._propagate_bits(new_path, phi)
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = [p for p in paths if crc_check(p.u_hat[info_mask], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
