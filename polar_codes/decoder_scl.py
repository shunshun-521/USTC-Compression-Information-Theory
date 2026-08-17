"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    f_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _lower_llr,
    _prepare_channel_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _compute_crc_bits(info_bits, crc_length, poly):
    crc = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        crc ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if crc & (1 << (crc_length - 1)):
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return [(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    crc_bits = _compute_crc_bits(info_bits, crc_length, poly)
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
    elif crc_length == 16:
        poly = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    expected = _compute_crc_bits(bits[:-crc_length], crc_length, poly)
    return np.array_equal(bits[-crc_length:], expected)


class Path:
    """单条译码路径（Lazy Copy）"""

    __slots__ = ('pm', 'u_hat', 'L', 'B', 'parent_L', 'parent_B')

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.parent_L = None
        self.parent_B = None

    def fork(self):
        child = Path(self.L.shape[1] - 1, len(self.u_hat))
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        child.parent_L = self.L
        child.parent_B = self.B
        child.L = None
        child.B = None
        return child

    def get_L(self):
        if self.L is None:
            self.L = self.parent_L.copy()
            self.parent_L = None
        return self.L

    def get_B(self):
        if self.B is None:
            self.B = self.parent_B.copy()
            self.parent_B = None
        return self.B


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llr(self, path, l):
        L = path.get_L()
        B = path.get_B()
        n, N = self.n, self.N

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1]) if not np.isnan(B[j - branch_size, s + 1]) else 0
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _propagate_bits(self, path, l, bit):
        B = path.get_B()
        n = self.n
        N = self.N
        B[l, n] = bit
        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        N, n = self.N, self.n

        root = Path(n, N)
        root.get_L()[:, 0] = llr_ch
        paths = [root]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                self._update_llr(path, l)
                L = path.get_L()
                llr = L[l, n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(llr, 0)
                    path.u_hat[l] = 0
                    self._propagate_bits(path, l, 0)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = path.fork()
                        child.pm += self._pm_penalty(llr, bit)
                        child.u_hat[l] = bit
                        self._propagate_bits(child, l, bit)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:self.list_size]

        best_path = min(paths, key=lambda p: p.pm)
        best_crc_path = None
        best_crc_pm = float('inf')

        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if path.pm < best_crc_pm:
                        best_crc_pm = path.pm
                        best_crc_path = path

        if best_crc_path is not None:
            return best_crc_path.u_hat.copy(), best_crc_pm
        return best_path.u_hat.copy(), best_path.pm
