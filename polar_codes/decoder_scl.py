"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode_channel


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位"""
    poly = _crc_poly(crc_length)
    reg = 0
    bits = np.asarray(info_bits, dtype=np.int8).ravel()
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([bits, np.array(crc_bits, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """校验 CRC"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("pm", "L", "C", "u")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.u = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.br = bit_reversal_permutation(N)
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]

    def _branch_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L_size == 1:
            u = sc_decode_channel(llr_ch, self.frozen.astype(int))
            return u.astype(int), 0.0
        paths = [_Path(self.n, self.N)]
        paths[0].L[self.n, :] = llr_ch[self.br]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                self._update_llr(path, phi)
                llr = path.L[0, phi]
                if self.frozen[phi]:
                    pen = self._branch_penalty(llr, 0)
                    path.pm += pen
                    path.u[phi] = 0
                    path.C[0, phi] = 0
                    self._update_bits(path, phi)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = self._clone_path(path)
                        child.pm += self._branch_penalty(llr, u)
                        child.u[phi] = u
                        child.C[0, phi] = u
                        self._update_bits(child, phi)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return best.u.astype(int), best.pm

    def _clone_path(self, path):
        q = _Path(self.n, self.N)
        q.pm = path.pm
        q.L = path.L.copy()
        q.C = path.C.copy()
        q.u = path.u.copy()
        return q

    def _update_llr(self, path, phi):
        for layer in range(self.n):
            psi = phi >> layer
            step = 1 << layer
            if psi % 2 == 0:
                block = (phi >> (layer + 1)) << (layer + 1)
                for j in range(step):
                    a = block + j
                    path.L[layer, a] = f_operation(
                        path.L[layer + 1, a], path.L[layer + 1, a + step]
                    )
            else:
                block = ((phi >> (layer + 1)) << (layer + 1)) + step
                for j in range(step):
                    a = block - step + j
                    path.L[layer, a] = g_operation(
                        path.L[layer + 1, a],
                        path.L[layer + 1, a + step],
                        path.C[layer, a],
                    )

    def _update_bits(self, path, phi):
        layer = 0
        while layer < self.n and (phi >> layer) % 2 == 0:
            block = (phi >> (layer + 1)) << (layer + 1)
            step = 1 << layer
            for j in range(step):
                a = block + j
                path.C[layer + 1, a] = (path.C[layer, a] ^ path.C[layer, a + step]) & 1
                path.C[layer + 1, a + step] = path.C[layer, a + step]
            layer += 1


def scl_equivalent_sc(llr_ch, frozen_bits):
    """L=1 的 SCL 应与 SC 一致（用于校验）"""
    u_scl, _ = SCLDecoder(len(llr_ch), frozen_bits, list_size=1).decode(llr_ch)
    u_sc = sc_decode_channel(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
