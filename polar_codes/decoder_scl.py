"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversed,
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length):
    """MSB-first CRC 余数（poly: 0x07 / 0x8005）。"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    msb_mask = 1 << (crc_length - 1)
    crc = 0
    for bit in bits:
        crc ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if crc & msb_mask:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后（MSB-first）。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    crc = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 crc_length 位是否为正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _pm_update(pm, llr, u):
    """路径度量更新：与 LLR 符号不一致时加 |LLR| 惩罚。"""
    hard = 0 if llr >= 0.0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _Path:
    """单条 SCL 路径（Lazy Copy）。"""

    __slots__ = ("L", "B", "pm", "parent", "branch_layer", "branch_index")

    def __init__(self, N, n, llr_ch, parent=None, branch_layer=-1, branch_index=-1):
        self.parent = parent
        self.branch_layer = branch_layer
        self.branch_index = branch_index
        if parent is None:
            self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=int)
            self.L[:, 0] = llr_ch
            self.pm = 0.0
        else:
            self.L = parent.L
            self.B = parent.B
            self.pm = parent.pm

    def fork(self):
        """创建共享 P/C 的子路径副本。"""
        child = _Path.__new__(_Path)
        child.L = self.L
        child.B = self.B
        child.pm = self.pm
        child.parent = self
        child.branch_layer = -1
        child.branch_index = -1
        return child

    def ensure_copy(self):
        """写入前复制 P/C（Lazy Copy）。"""
        if self.parent is not None:
            self.L = self.L.copy()
            self.B = self.B.copy()
            p = self.parent
            while p is not None and p.parent is not None:
                p = p.parent
            self.parent = None


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, l):
        start = self.n - active_llr_level(l, self.n)
        for s in range(start, self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = path.L[j, s]
                    btm_llr = path.L[j + branch_size, s]
                    path.L[j, s + 1] = f_operation(top_llr, btm_llr)
                else:
                    btm_llr = path.L[j, s]
                    top_llr = path.L[j - branch_size, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        end = self.n - active_bit_level(l, self.n)
        for s in range(self.n, end, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            pm: 最优路径度量
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for idx in range(self.N):
            l = bit_reversed(idx, self.n)
            candidates = []

            for path in paths:
                path.ensure_copy()
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_pm = _pm_update(path.pm, llr, 0)
                    path.pm = new_pm
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        child = path.fork()
                        child.ensure_copy()
                        child.pm = _pm_update(path.pm, llr, u)
                        child.B[l, self.n] = u
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p.pm)

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            for path in paths:
                payload = path.B[info_positions, self.n]
                if crc_check(payload, self.crc_length):
                    return path.B[:, self.n].astype(int), path.pm

        best = paths[0]
        return best.B[:, self.n].astype(int), best.pm
