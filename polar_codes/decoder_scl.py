"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, precompute_sc_indices,
    _active_llr_level, _active_bit_level, _bit_reversed,
)

CRC8_POLY = 0x107
CRC16_POLY = 0x11021


def _crc_poly(crc_length):
    return CRC8_POLY if crc_length == 8 else CRC16_POLY


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    n = len(info_bits)
    msg = 0
    for b in info_bits:
        msg = (msg << 1) | int(b)
    msg <<= crc_length
    for i in range(n + crc_length - 1, crc_length - 1, -1):
        if (msg >> i) & 1:
            msg ^= poly << (i - crc_length)
    crc_bits = np.array(
        [(msg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    msg = 0
    for b in bits:
        msg = (msg << 1) | int(b)
    n = len(bits)
    for i in range(n - 1, crc_length - 1, -1):
        if (msg >> i) & 1:
            msg ^= poly << (i - crc_length)
    return msg == 0


class PathState:
    """单条 SCL 路径状态。"""

    __slots__ = ('pm', 'L', 'B', 'u_hat')

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _compute_llrs(self, phi, path):
        l = self.decode_order[phi]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            half = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], path.B[j - half, s + 1]
                    )
        return l, path.L[l, self.n]

    def _update_path_bits(self, phi, l, path):
        path.B[l, self.n] = path.u_hat[l]
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                half = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= half:
                        path.B[j - half, s - 1] = path.B[j, s] ^ path.B[j - half, s]
                        path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, path):
        new_path = PathState(self.N, self.n)
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                l, llr = self._compute_llrs(phi, path)

                if self.frozen_bits[l]:
                    path.pm += self._path_metric_penalty(llr, 0)
                    path.u_hat[l] = 0
                    self._update_path_bits(phi, l, path)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child.pm += self._path_metric_penalty(llr, bit)
                        child.u_hat[l] = bit
                        self._update_path_bits(phi, l, child)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
