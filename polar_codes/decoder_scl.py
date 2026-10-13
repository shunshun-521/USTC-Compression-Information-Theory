"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, precompute_sc_indices


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_lfsr(bits, crc_length, poly):
    """按位 LFSR（MSB 优先），用于 CRC-8 / CRC-16"""
    mask = (1 << crc_length) - 1
    reg = 0
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = _crc_lfsr(info_bits, crc_length, poly)
    crc_bits = np.zeros(crc_length, dtype=np.int8)
    for i in range(crc_length):
        crc_bits[i] = (reg >> (crc_length - 1 - i)) & 1
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = _crc_lfsr(bits, crc_length, poly)
    return reg == 0


class _Path:
    __slots__ = ("P", "C", "pm", "u_hat", "parent")

    def __init__(self, n, N):
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.parent = None

    def fork(self):
        child = _Path(self.P.shape[0] - 1, self.P.shape[1])
        child.P = self.P.copy()
        child.C = self.C.copy()
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        child.parent = self
        return child


class SCLDecoder:
    """SCL 译码器（路径复制在分裂时进行）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _path_metric_penalty(self, llr, u):
        """与 SC 一致：不一致分支加 |LLR|"""
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode

            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.n, self.N)]
        paths[0].P[self.n, :] = llr_ch

        for phi in range(self.N):
            new_candidates = []

            for path in paths:
                self._update_llr(path, phi)
                llr0 = path.P[0, 0]

                if self.frozen_bits[phi]:
                    pen = self._path_metric_penalty(llr0, 0)
                    path.pm += pen
                    path.u_hat[phi] = 0
                    path.C[0, 0] = 0
                    self._update_bits(path, phi)
                    new_candidates.append(path)
                else:
                    for u in (0, 1):
                        child = path.fork()
                        pen = self._path_metric_penalty(llr0, u)
                        child.pm += pen
                        child.u_hat[phi] = u
                        child.C[0, 0] = u
                        self._update_bits(child, phi)
                        new_candidates.append(child)

            new_candidates.sort(key=lambda p: p.pm)
            paths = new_candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            for p in paths:
                payload = p.u_hat[info_mask]
                if crc_check(payload, self.crc_length):
                    return p.u_hat.copy(), p.pm
        best = paths[0]
        return best.u_hat.copy(), best.pm

    def _update_llr(self, path, phi):
        l = 0
        while (phi >> l) & 1:
            l += 1
        n = self.n
        P = path.P
        C = path.C
        lambda_offset = self.lambda_offset

        for layer in range(l, n):
            psi = phi >> (layer + 1)
            pm = psi << (layer + 1)
            span = lambda_offset[layer]
            La = P[layer + 1, pm : pm + span]
            Lb = P[layer + 1, pm + span : pm + 2 * span]
            P[layer, pm : pm + span] = f_operation(La, Lb)

        for layer in range(n - 1, l - 1, -1):
            psi = phi >> (layer + 1)
            pm = psi << (layer + 1)
            span = lambda_offset[layer]
            for beta in range(span):
                C[layer, pm + beta] = C[layer + 1, pm + beta]
                P[layer, pm + beta] = g_operation(
                    P[layer + 1, pm + beta],
                    P[layer + 1, pm + beta + span],
                    C[layer, pm + beta],
                )

    def _update_bits(self, path, phi):
        C = path.C
        for layer in self.bit_layer_vec[phi]:
            psi = phi >> (layer + 1)
            pm = psi << (layer + 1)
            span = self.lambda_offset[layer]
            for beta in range(span):
                C[layer + 1, pm + beta + span] = C[layer, pm + 2 * beta + 1]
                C[layer + 1, pm + beta] = (
                    C[layer, pm + 2 * beta] ^ C[layer, pm + 2 * beta + 1]
                )
