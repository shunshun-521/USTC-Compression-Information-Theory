"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    upper_llr, lower_llr, active_llr_level, active_bit_level, logdomain_sum
)


# CRC 多项式
CRC_POLYS = {
    8: 0x07,      # x^8 + x^2 + x + 1
    16: 0x8005,   # CRC-16
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class PathState:
    """单条译码路径状态"""
    __slots__ = ('L', 'B', 'pm', 'u_hat', 'active')

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch.copy()
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, paths, l):
        for path in paths:
            if not path.active:
                continue
            for s in range(self.n - active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                    else:
                        path.L[j, s + 1] = lower_llr(
                            path.L[j, s], path.L[j - branch_size, s], path.B[j - branch_size, s + 1]
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            if not path.active:
                continue
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                        path.B[j, s - 1] = path.B[j, s]

    def _copy_path(self, src):
        """Lazy copy：复制路径状态"""
        dst = PathState.__new__(PathState)
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        dst.active = True
        return dst

    def _pm_penalty(self, llr, bit):
        """路径度量惩罚"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            self._update_llrs(paths, l)

            current_llr = paths[0].L[l, self.n] if paths else 0.0
            new_paths = []

            if self.frozen_bits[l]:
                for path in paths:
                    if not path.active:
                        continue
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    path.pm += self._pm_penalty(path.L[l, self.n], 0)
                    new_paths.append(path)
            else:
                for path in paths:
                    if not path.active:
                        continue
                    llr_val = path.L[l, self.n]
                    for bit in (0, 1):
                        p = self._copy_path(path)
                        p.u_hat[l] = bit
                        p.B[l, self.n] = bit
                        p.pm += self._pm_penalty(llr_val, bit)
                        new_paths.append(p)

            # 路径裁剪
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

            self._update_bits(paths, l)

        # 选择最优路径
        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)
            else:
                best = min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
