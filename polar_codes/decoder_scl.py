"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07 (x^8 + x^2 + x + 1)
    CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    crc = 0

    for bit in info_bits:
        crc ^= int(bit) << (crc_length - 1)
        if crc & (1 << (crc_length - 1)):
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask

    crc_bits = np.array(
        [(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = 0x07 if crc_length == 8 else 0x8005
    mask = (1 << crc_length) - 1
    crc = 0
    for bit in bits:
        crc ^= int(bit) << (crc_length - 1)
        if crc & (1 << (crc_length - 1)):
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    return crc == 0


class Path:
    """单条 SCL 译码路径。"""

    def __init__(self, N, n):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]
        self.llr_layers = []
        self.bit_layers = []
        for l in self.decode_order:
            start_llr = self.n - active_llr_level(l, self.n)
            self.llr_layers.append(list(range(start_llr, self.n)))
            if l < N / 2:
                self.bit_layers.append([])
            else:
                start_bit = self.n - active_bit_level(l, self.n)
                self.bit_layers.append(list(range(self.n, start_bit, -1)))

    def _update_llrs(self, path, l, layer_idx):
        for s in self.llr_layers[layer_idx]:
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l, layer_idx):
        for s in self.bit_layers[layer_idx]:
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        """路径度量惩罚：判决与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat（最优路径估计），pm（路径度量）
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for layer_idx, l in enumerate(self.decode_order):
            new_paths = []

            for path in paths:
                self._update_llrs(path, l, layer_idx)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    self._update_bits(path, l, layer_idx)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = Path(self.N, self.n)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = path.pm + self._pm_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        self._update_bits(child, l, layer_idx)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        # 选择最优路径
        crc_pass = []
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)
            else:
                crc_pass.append(path)

        if crc_pass:
            best = min(crc_pass, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm
