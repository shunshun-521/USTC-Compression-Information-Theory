"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, _b_check, _s_updater, _li

_INF = np.inf

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _poly_bits(poly, length):
    """将多项式整数转为比特数组（高位在前）"""
    bits = []
    for i in range(length - 1, -1, -1):
        bits.append((poly >> i) & 1)
    return np.array(bits, dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    使用标准多项式：r=8 CRC-8 (0x07)，r=16 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    poly_bits = _poly_bits(poly, crc_length + 1)
    reg = np.zeros(crc_length, dtype=int)
    for bit in info_bits:
        msb = reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if msb ^ bit:
            reg ^= poly_bits[1:]
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class _PathState:
    """单条 SCL 路径状态"""

    __slots__ = ("llrs", "s", "pm", "u_hat")

    def __init__(self, n, N):
        self.llrs = np.full((n + 1, N), _INF, dtype=np.float64)
        self.s = np.full((n + 1, N), -1, dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        n = self.llrs.shape[0] - 1
        N = self.llrs.shape[1]
        p = _PathState(n, N)
        p.llrs = self.llrs.copy()
        p.s = self.s.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _init_paths(self, llr_ch):
        path = _PathState(self.n, self.N)
        path.llrs[self.n, :] = llr_ch
        return [path]

    def _compute_llr(self, path, phi):
        return _li(0, phi, path.llrs, path.s)

    def _update_path_bit(self, path, phi, bit):
        path.u_hat[phi] = bit
        path.s[0, phi] = bit

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            pm: 最优路径的度量值
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._init_paths(llr_ch)

        for phi in range(self.N):
            candidates = []

            for path in paths:
                llr_val = self._compute_llr(path, phi)

                if self.frozen_bits[phi]:
                    new_path = path.copy()
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    new_path.pm += penalty
                    self._update_path_bit(new_path, phi, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        consistent = (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0)
                        if not consistent:
                            new_path.pm += abs(llr_val)
                        self._update_path_bit(new_path, phi, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_path = self._select_best_path(paths)
        return best_path.u_hat, best_path.pm

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                return min(crc_pass, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
