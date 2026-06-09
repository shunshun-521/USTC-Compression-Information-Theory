"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    f_operation,
    g_operation,
)


_CRC_POLY_LOC = {
    8: [8, 2, 1, 0],
    16: [16, 15, 2, 0],
}


def _crc_poly(crc_length):
    loc = _CRC_POLY_LOC[crc_length]
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def _crc_division(info_bits, crc_length):
    """GF(2) 多项式除法求 CRC 余数。"""
    p = _crc_poly(crc_length)
    info = [int(b) for b in info_bits]
    times = len(info)
    for _ in range(crc_length):
        info.append(0)
    for i in range(times):
        if info[i] == 1:
            for j in range(crc_length + 1):
                info[j + i] ^= p[j]
    return np.array(info[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    check = _crc_division(info_bits, crc_length)
    return np.concatenate([info_bits, check])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info_len = len(bits) - crc_length
    recoded = crc_encode(bits[:info_len], crc_length)
    return np.array_equal(recoded, bits)


def _path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


def _update_llrs_path(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], top_bit
                )


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 P/C 状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            is_frozen = self.frozen_bits[l]
            new_paths = []

            for path in paths:
                _update_llrs_path(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if is_frozen:
                    bit = 0
                    path.pm += _path_metric_penalty(llr, bit)
                    path.u_hat[l] = bit
                    path.B[l, self.n] = bit
                    _update_bits(path.B, l, self.n)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, llr_ch)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = path.pm + _path_metric_penalty(llr, bit)
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat.copy(), path.pm

        best = paths[0]
        return best.u_hat.copy(), best.pm
