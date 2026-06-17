"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    _update_llr_layer,
    _update_bit_layer,
    _remap_channel_llrs,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    payload = bits[:-crc_length]
    expected = _crc_remainder(payload, poly, crc_length)
    received = 0
    for i in range(crc_length):
        received = (received << 1) | int(bits[-crc_length + i])
    return expected == received


class Path:
    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _copy_path(self, src, dst):
        dst.pm = src.pm
        dst.L[:] = src.L
        dst.B[:] = src.B
        dst.u_hat[:] = src.u_hat

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = _remap_channel_llrs(llr_ch)
        N, n, L = self.N, self.n, self.list_size

        paths = [Path(n, N)]
        paths[0].L[:, 0] = llr_ch

        for phi, l in enumerate(self.decode_order):
            candidates = []

            for path in paths:
                for layer in self.llr_layer_vec[phi]:
                    _update_llr_layer(path.L, path.B, layer, l, n, N)

                llr = path.L[l, n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, n] = 0
                    path.u_hat[l] = 0
                    for layer in self.bit_layer_vec[phi]:
                        _update_bit_layer(path.B, layer, l, n, N)
                    candidates.append((path.pm, path))
                else:
                    for bit in (0, 1):
                        child = Path(n, N)
                        self._copy_path(path, child)
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.B[l, n] = bit
                        child.u_hat[l] = bit
                        for layer in self.bit_layer_vec[phi]:
                            _update_bit_layer(child.B, layer, l, n, N)
                        candidates.append((child.pm, child))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[:L]]

        best_path = min(paths, key=lambda p: p.pm)

        if self.crc_length > 0:
            crc_paths = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            if crc_paths:
                best_path = min(crc_paths, key=lambda p: p.pm)

        return best_path.u_hat.copy(), best_path.pm
