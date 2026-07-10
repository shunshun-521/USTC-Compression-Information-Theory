"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, precompute_sc_indices, _prepare_frozen


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= poly
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    __slots__ = ("P", "C", "pm", "u_hat")

    def __init__(self, N, n):
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _prepare_frozen(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)
        self.br = bit_reversal_permutation(N)
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src):
        dst = _Path(self.N, self.n)
        dst.P[:] = src.P
        dst.C[:] = src.C
        dst.pm = src.pm
        dst.u_hat[:] = src.u_hat
        return dst

    def _update_llr_layers(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            sp = self.lambda_offset[layer]
            P, C = path.P, path.C
            for beta in range(0, self.N, 2 * sp):
                for omega in range(sp):
                    idx = beta + omega
                    P[layer, idx] = f_operation(P[layer + 1, idx], P[layer + 1, idx + sp])
                    P[layer, idx + sp] = g_operation(
                        P[layer + 1, idx], P[layer + 1, idx + sp], C[layer, idx]
                    )

    def _update_bit_layers(self, path, phi):
        for layer in self.bit_layer_vec[phi]:
            sp = self.lambda_offset[layer]
            C = path.C
            for beta in range(0, self.N, 2 * sp):
                for omega in range(sp):
                    idx = beta + omega
                    C[layer + 1, idx] = (C[layer, idx] + C[layer, idx + sp]) % 2
                    C[layer + 1, idx + sp] = C[layer, idx + sp]

    def _path_metric_penalty(self, llr_val, u_bit):
        u_hd = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == u_hd else abs(llr_val)

    def _crc_passes(self, u_hat):
        if self.crc_length <= 0:
            return True
        info_bits = u_hat[self.info_positions]
        return crc_check(info_bits, self.crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br].copy()

        paths = [_Path(self.N, self.n)]
        paths[0].P[self.n, : self.N] = llr

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                self._update_llr_layers(path, phi)
                llr_bit = path.P[0, phi]

                if self.frozen_bits[phi]:
                    path.pm += self._path_metric_penalty(llr_bit, 0)
                    path.u_hat[phi] = 0
                    path.C[0, phi] = 0
                    self._update_bit_layers(path, phi)
                    new_paths.append(path)
                else:
                    p0 = path
                    p1 = self._copy_path(path)
                    for child, bit in ((p0, 0), (p1, 1)):
                        child.pm += self._path_metric_penalty(llr_bit, bit)
                        child.u_hat[phi] = bit
                        child.C[0, phi] = bit
                        self._update_bit_layers(child, phi)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_paths = [p for p in paths if self._crc_passes(p.u_hat)]
        best = min(crc_paths if crc_paths else paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
