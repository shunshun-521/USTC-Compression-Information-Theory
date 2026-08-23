"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed_index
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(data_bits, crc_length):
    """计算 CRC 余数位（MSB 优先）"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for bit in data_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07), r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = _crc_remainder(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class _Path:
    """SCL 单条路径状态"""

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（路径复制实现，L 较小时足够高效）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], path.B[j - half, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 2 ** s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    path.B[j - half, s - 1] = int(path.B[j, s]) ^ int(path.B[j - half, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        preferred = 0 if llr >= 0 else 1
        return 0.0 if u == preferred else abs(llr)

    def decode(self, llr_ch):
        """
        返回 (u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = []
        p0 = _Path(self.N, self.n)
        p0.L[:, 0] = llr_ch
        paths.append(p0)
        u_hat_full = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            l = bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    u_hat_full[l] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        new_path = _Path(self.N, self.n)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = path.pm + self._pm_penalty(llr, u)
                        new_path.B[l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        # 选择最优路径
        best = None
        if self.crc_length > 0:
            crc_ok_paths = []
            for path in paths:
                u = path.B[:, self.n].astype(int)
                info_bits = u[np.where(self.frozen_bits == 0)[0]]
                if crc_check(info_bits, self.crc_length):
                    crc_ok_paths.append(path)
            if crc_ok_paths:
                best = min(crc_ok_paths, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        u_hat = best.B[:, self.n].astype(int)
        return u_hat, best.pm
