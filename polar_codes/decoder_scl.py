"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    sc_decode,
)
from encoder import prepare_channel_llr


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, crc_length, poly):
    """逐位 CRC 处理，返回最终寄存器值"""
    crc = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for bit in bits:
        crc ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if crc & msb:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, crc_length, poly)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class PathState:
    """单条 SCL 路径状态（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float32)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float32)
        self.L[:, 0] = llr_ch.astype(np.float32)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        new = PathState.__new__(PathState)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new

    def _update_llrs(self, l, n):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.L.shape[0], block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        int(self.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, l, n, N):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]

    def current_llr(self, l, n):
        self._update_llrs(l, n)
        return self.L[l, n]


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_adj = prepare_channel_llr(llr_ch, self.N)
        paths = [PathState(self.N, self.n, llr_adj)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                llr_val = path.current_llr(l, self.n)

                if self.frozen_bits[l]:
                    child = path.copy()
                    child.pm = self._pm_update(child.pm, llr_val, 0)
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    child._update_bits(l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm = self._pm_update(child.pm, llr_val, u)
                        child.u_hat[l] = u
                        child.B[l, self.n] = u
                        child._update_bits(l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = None
        best_path = None

        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path
            if best_pm is None or path.pm < best_pm:
                best_pm = path.pm
                best_path = path

        chosen = best_crc if best_crc is not None else best_path
        return chosen.u_hat.astype(int).copy(), chosen.pm
