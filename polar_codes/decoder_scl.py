"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed_index
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level, _update_llrs, _update_bits


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制 + 列表裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr_val, bit):
        u_from_llr = 0 if llr_val >= 0 else 1
        return 0.0 if bit == u_from_llr else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr_leaf = path.L[l, self.n]
                if l in self.frozen_set:
                    bit = 0
                    path.pm += self._pm_penalty(llr_leaf, 0)
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n)
                    path.u_hat[l] = 0
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = _Path(self.N, self.n, llr_ch)
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.pm = path.pm + self._pm_penalty(llr_leaf, bit)
                        p.u_hat = path.u_hat.copy()
                        p.B[l, self.n] = bit
                        _update_bits(p.B, l, self.n)
                        p.u_hat[l] = bit
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            info_idx = np.sort(info_idx)
            valid = [p for p in paths if crc_check(p.u_hat[info_idx], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
        return best.u_hat, best.pm
