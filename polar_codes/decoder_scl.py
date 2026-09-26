"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    align_llr_to_decoder,
    f_operation,
    g_operation,
    precompute_sc_indices,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.uint8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.uint8)
    if len(bits) < crc_length:
        return False
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(recomputed, bits)


class _Path:
    __slots__ = ("pm", "u_hat", "P", "C", "parent", "phi_done")

    def __init__(self, N, n, lambda_offset, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.P = np.zeros(2 * N, dtype=np.float64)
        self.P[lambda_offset[n] : lambda_offset[n] + N] = llr_ch.copy()
        self.C = np.zeros((2 * N - 1, 2), dtype=int)
        self.parent = None
        self.phi_done = -1


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 P/C）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(
            N
        )

    def _advance_llr(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            psi = phi // (1 << layer)
            if psi % 2 == 0:
                off = self.lambda_offset[layer]
                span = 1 << (self.n - layer)
                for beta in range(span):
                    i = off + beta
                    path.P[i] = f_operation(path.P[i], path.P[i + span])
            else:
                off = self.lambda_offset[layer]
                span = 1 << (self.n - layer)
                for beta in range(span):
                    i = off + beta
                    u_val = path.C[2 * i + 1, phi % 2]
                    path.P[i] = g_operation(path.P[i], path.P[i + span], u_val)

    def _bit_backtrack(self, path, phi, u_bit):
        path.C[2 * self.lambda_offset[0], phi % 2] = u_bit
        for layer in self.bit_layer_vec[phi]:
            off = self.lambda_offset[layer]
            span = 1 << (self.n - layer)
            for beta in range(span):
                i = off + beta
                path.C[2 * i, (phi + 1) % 2] = (
                    path.C[2 * i, phi % 2] ^ path.C[2 * i + 1, phi % 2]
                )
                path.C[2 * i + 1, (phi + 1) % 2] = path.C[2 * i + 1, phi % 2]

    def _copy_path(self, src):
        dst = _Path(self.N, self.n, self.lambda_offset, np.zeros(self.N))
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        dst.P = src.P.copy()
        dst.C = src.C.copy()
        dst.parent = src
        dst.phi_done = src.phi_done
        return dst

    def decode(self, llr_ch):
        llr_ch = align_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
        if self.list_size == 1 and self.crc_length == 0:
            from native_sc_bridge import native_sc_decode

            u_hat = native_sc_decode(llr_ch, self.frozen_bits.astype(int))
            return u_hat, 0.0
        paths = [_Path(self.N, self.n, self.lambda_offset, llr_ch)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                if path.phi_done >= phi:
                    new_paths.append(path)
                    continue
                self._advance_llr(path, phi)
                llr0 = path.P[self.lambda_offset[0]]

                if self.frozen_bits[phi]:
                    u = 0
                    pm = path.pm + (abs(llr0) if llr0 < 0 else 0.0)
                    p = self._copy_path(path)
                    p.pm = pm
                    p.u_hat[phi] = u
                    self._bit_backtrack(p, phi, u)
                    p.phi_done = phi
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        llr_sign = 0 if llr0 >= 0 else 1
                        pm = path.pm + (0.0 if u == llr_sign else abs(llr0))
                        p = self._copy_path(path)
                        p.pm = pm
                        p.u_hat[phi] = u
                        self._bit_backtrack(p, phi, u)
                        p.phi_done = phi
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
