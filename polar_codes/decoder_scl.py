"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, precompute_sc_indices


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(recomputed, bits)


class _PathState:
    __slots__ = ("P", "C", "pm", "u_hat", "parent", "active")

    def __init__(self, N, n, lambda_offset, llr_ch):
        self.P = [np.zeros(lambda_offset[l], dtype=np.float64) for l in range(n + 1)]
        self.C = [np.zeros(lambda_offset[l], dtype=int) for l in range(n + 1)]
        self.P[n][:] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.parent = None
        self.active = True


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
        self._info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, phi):
        for layer in self.llr_layer_vec[phi]:
            lam = self.lambda_offset[layer]
            half = lam // 2
            for beta in range(half):
                path.P[layer][beta] = f_operation(
                    path.P[layer + 1][2 * beta], path.P[layer + 1][2 * beta + 1]
                )
                path.C[layer][beta] = (
                    path.C[layer + 1][2 * beta] ^ path.C[layer + 1][2 * beta + 1]
                )
                path.P[layer][beta + half] = g_operation(
                    path.P[layer + 1][2 * beta],
                    path.P[layer + 1][2 * beta + 1],
                    path.C[layer][beta],
                )

    def _update_bits(self, path, phi):
        for layer in self.bit_layer_vec[phi]:
            lam = self.lambda_offset[layer]
            half = lam // 2
            for beta in range(half):
                path.C[layer + 1][2 * beta] = (
                    path.C[layer][beta] ^ path.C[layer][beta + half]
                )
                path.C[layer + 1][2 * beta + 1] = path.C[layer][beta + half]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode

            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_PathState(self.N, self.n, self.lambda_offset, llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                self._update_llrs(path, phi)
                llr0 = path.P[0][0]

                if self.frozen_bits[phi]:
                    u = 0
                    new_path = path
                    new_path.pm += self._pm_penalty(llr0, u)
                    new_path.u_hat[phi] = u
                    new_path.C[0][0] = u
                    self._update_bits(new_path, phi)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        if u == 0:
                            new_path = path
                        else:
                            new_path = _PathState(
                                self.N, self.n, self.lambda_offset, llr_ch
                            )
                            for l in range(self.n + 1):
                                new_path.P[l][:] = path.P[l]
                                new_path.C[l][:] = path.C[l]
                            new_path.pm = path.pm
                            new_path.u_hat[:] = path.u_hat
                        new_path.pm += self._pm_penalty(llr0, u)
                        new_path.u_hat[phi] = u
                        new_path.C[0][0] = u
                        self._update_bits(new_path, phi)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                payload = p.u_hat[self._info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
