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
    sc_decode,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
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
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


def _path_metric_update(pm, llr, u):
    """路径度量更新：与 LLR 符号不一致时加 |LLR|。"""
    u_from_llr = 0 if llr >= 0 else 1
    if u != u_from_llr:
        return pm + abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "parent", "branch_bit", "branch_layer")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch.copy()
        self.pm = 0.0
        self.parent = None
        self.branch_bit = None
        self.branch_layer = None


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phase in range(self.N):
            l = bit_reversed(phase, self.n)
            new_candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pm = _path_metric_update(path.pm, llr, 0)
                    child = self._clone_path(path)
                    child.pm = pm
                    self._set_bit(child, l, 0)
                    new_candidates.append(child)
                else:
                    for u in (0, 1):
                        pm = _path_metric_update(path.pm, llr, u)
                        child = self._clone_path(path)
                        child.pm = pm
                        self._set_bit(child, l, u)
                        new_candidates.append(child)

            new_candidates.sort(key=lambda p: p.pm)
            paths = new_candidates[: self.list_size]

        crc_paths = []
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.B[self.info_indices, self.n]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(path)

        best = min(crc_paths or paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm

    def _clone_path(self, path):
        child = _Path(self.N, self.n, path.L[:, 0])
        child.L = path.L.copy()
        child.B = path.B.copy()
        child.pm = path.pm
        return child

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
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
                    path.L[j, s + 1] = g_operation(btm_llr, top_llr, top_bit)

    def _set_bit(self, path, l, u):
        path.B[l, self.n] = u
        if l >= self.N // 2:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                        )
                        path.B[j, s - 1] = path.B[j, s]
