"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
    _bit_rev,
    _update_llrs,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
    """按位长除法计算 CRC 校验位。"""
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_division(np.asarray(info_bits, dtype=int), poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    data = np.asarray(bits, dtype=int)
    expected = _crc_division(data[:-crc_length], poly, crc_length)
    return np.array_equal(data[-crc_length:], expected)


class _Path:
    """单条 SCL 路径。"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        brp = bit_reversal_permutation(N)
        self.L[:, 0] = llr_ch[brp]
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr_val, bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if hard == bit else abs(llr_val)

    def _advance_path(self, path, l):
        """对单条路径执行一步 SC 更新并返回当前 LLR。"""
        _update_llrs(path.L, path.B, l, self.n)
        return path.L[l, self.n]

    def _set_bit_and_backprop(self, path, l, bit):
        path.B[l, self.n] = bit
        path.u_hat[l] = bit
        _update_bits(path.B, l, self.n, self.N)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]
        decode_order = [_bit_rev(i, self.n) for i in range(self.N)]

        for l in decode_order:
            new_paths = []
            for path in paths:
                llr_val = self._advance_path(path, l)

                if l in self.frozen_set:
                    penalty = self._path_metric_penalty(llr_val, 0)
                    new_path = self._clone_path(path)
                    self._set_bit_and_backprop(new_path, l, 0)
                    new_path.pm += penalty
                    new_paths.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._clone_path(path)
                        self._set_bit_and_backprop(new_path, l, bit)
                        new_path.pm += self._path_metric_penalty(llr_val, bit)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_ok = [p for p in paths if self._crc_passes(p)]
            if crc_ok:
                paths = crc_ok

        best = paths[0]
        return best.u_hat.copy(), best.pm

    def _clone_path(self, path):
        """Lazy copy：仅在需要时复制数组。"""
        new_path = _Path.__new__(_Path)
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _crc_passes(self, path):
        info_bits = path.u_hat[self.info_indices]
        return crc_check(info_bits, self.crc_length)
