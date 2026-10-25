"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, precompute_sc_indices, sc_decode_recursive

# CRC-8: 0x07, CRC-16: 0x8005
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    """按比特 MSB 优先计算 CRC 余数。"""
    mask = (1 << crc_len) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        for _ in range(1):
            if reg & (1 << (crc_len - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("P", "C", "pm", "u_hat", "parent", "active")

    def __init__(self, n, N):
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.parent = None
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时浅拷贝 P/C）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        _, self.llr_layers, self.bit_layers = precompute_sc_indices(N)

    def _llr_at_phi(self, path, phi):
        """计算路径在比特 phi 的叶节点 LLR（增量更新 P）。"""
        P, C = path.P, path.C
        n, N = self.n, self.N
        if phi == 0:
            for layer in range(n - 1, -1, -1):
                span = 1 << (n - layer)
                half = span // 2
                for base in range(0, N, span):
                    for j in range(half):
                        i = base + j
                        i2 = i + half
                        P[layer, i] = f_operation(P[layer + 1, i], P[layer + 1, i2])
                        P[layer, i2] = g_operation(
                            P[layer + 1, i], P[layer + 1, i2], C[layer + 1, i]
                        )
        else:
            for layer in self.llr_layers[phi]:
                span = 1 << (n - layer)
                half = span // 2
                base = (phi // span) * span
                off = phi % span
                if off < half:
                    i = base + off
                    i2 = i + half
                    P[layer, i] = f_operation(P[layer + 1, i], P[layer + 1, i2])
                else:
                    i = base + off
                    i_left = i - half
                    P[layer, i] = g_operation(
                        P[layer + 1, i_left], P[layer + 1, i], C[layer + 1, i_left]
                    )
        return P[0, phi]

    def _bit_propagate(self, path, phi, u_bit):
        C = path.C
        C[0, phi] = u_bit
        for layer in self.bit_layers[phi]:
            span = 1 << layer
            base = (phi // (2 * span)) * (2 * span)
            off = phi % span
            i = base + off
            C[layer + 1, i] = (C[layer, i] ^ C[layer, i + span]) % 2
            C[layer + 1, i + span] = C[layer, i + span]

    def _pm_penalty(self, llr, u_bit):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        n, N = self.n, self.N

        root = _Path(n, N)
        root.P[n, :] = llr_ch
        paths = [root]

        for phi in range(N):
            candidates = []
            for path in paths:
                if not path.active:
                    continue
                llr = self._llr_at_phi(path, phi)
                if self.frozen_bits[phi]:
                    pen = self._pm_penalty(llr, 0)
                    new_p = _Path(n, N)
                    new_p.P = path.P.copy()
                    new_p.C = path.C.copy()
                    new_p.u_hat = path.u_hat.copy()
                    new_p.pm = path.pm + pen
                    new_p.u_hat[phi] = 0
                    self._bit_propagate(new_p, phi, 0)
                    candidates.append(new_p)
                else:
                    for u_bit in (0, 1):
                        pen = self._pm_penalty(llr, u_bit)
                        new_p = _Path(n, N)
                        new_p.P = path.P.copy()
                        new_p.C = path.C.copy()
                        new_p.u_hat = path.u_hat.copy()
                        new_p.pm = path.pm + pen
                        new_p.u_hat[phi] = u_bit
                        self._bit_propagate(new_p, phi, u_bit)
                        candidates.append(new_p)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        # 选择最优路径
        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            crc_ok = []
            for p in paths:
                if crc_check(p.u_hat[info_idx], self.crc_length):
                    crc_ok.append(p)
            if crc_ok:
                best = min(crc_ok, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
