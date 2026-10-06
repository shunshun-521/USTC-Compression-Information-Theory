"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class Path:
    __slots__ = ("pm", "u_hat", "P", "C", "active")

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 P/C）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        _, self.llr_layer_vec, self.bit_layer_vec = self._precompute()
        self.lambda_offset = [1 << i for i in range(self.n + 1)]

    def _precompute(self):
        from decoder_sc import precompute_sc_indices

        return precompute_sc_indices(self.N)

    def _copy_path(self, src):
        p = Path(self.n, self.N)
        p.pm = src.pm
        p.u_hat = src.u_hat.copy()
        p.P = src.P.copy()
        p.C = src.C.copy()
        return p

    def _update_llr_layers(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            psi = (phi // self.lambda_offset[layer + 1]) * self.lambda_offset[layer + 1]
            for omega in range(self.lambda_offset[layer]):
                idx = psi + omega
                if layer == self.n - 1:
                    la = path.P[self.n, idx]
                    lb = path.P[self.n, idx + self.lambda_offset[layer]]
                else:
                    la = path.P[layer + 1, idx]
                    lb = path.P[layer + 1, idx + self.lambda_offset[layer]]
                path.P[layer, idx] = f_operation(la, lb)
                path.P[layer, idx + self.lambda_offset[layer]] = g_operation(
                    la, lb, path.C[layer, idx]
                )

    def _propagate_bits(self, path, phi):
        path.C[0, 0] = path.u_hat[phi]
        for layer in self.bit_layer_vec[phi]:
            psi = (phi // self.lambda_offset[layer + 1]) * self.lambda_offset[
                layer + 1
            ]
            for omega in range(self.lambda_offset[layer]):
                idx = psi + omega
                path.C[layer + 1, idx] = (
                    path.C[layer, idx] ^ path.C[layer, idx + self.lambda_offset[layer]]
                )
                path.C[layer + 1, idx + self.lambda_offset[layer]] = path.C[
                    layer, idx + self.lambda_offset[layer]
                ]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        if self.list_size == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        llr_ch = llr_ch[bit_reversal_permutation(N)]

        paths = [Path(self.n, N)]
        paths[0].P[self.n, :] = llr_ch

        for phi in range(N):
            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                self._update_llr_layers(path, phi)
                llr = path.P[0, 0]

                if self.frozen_bits[phi]:
                    path.u_hat[phi] = 0
                    path.pm += self._pm_penalty(llr, 0)
                    self._propagate_bits(path, phi)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        p = self._copy_path(path)
                        p.u_hat[phi] = u_bit
                        p.pm += self._pm_penalty(llr, u_bit)
                        self._propagate_bits(p, phi)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        info_mask = ~self.frozen_bits
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[info_mask]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
