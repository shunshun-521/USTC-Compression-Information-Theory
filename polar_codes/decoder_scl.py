"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _SCDEngine,
    _bit_reversed,
    _prepare_llr,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        msb = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if msb:
            reg ^= poly
    return reg == 0


class Path:
    __slots__ = ('pm', 'engine')

    def __init__(self, N, frozen_bits, llr):
        self.pm = 0.0
        self.engine = _SCDEngine(N, frozen_bits, llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数"""
        llr = _prepare_llr(llr_ch)
        paths = [Path(self.N, self.frozen_bits, llr)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            is_frozen = l in self.frozen_set
            new_paths = []

            for path in paths:
                path.engine._update_llrs(l)
                current_llr = path.engine.L[l, self.n]

                if is_frozen:
                    if current_llr < 0:
                        path.pm += abs(current_llr)
                    path.engine.B[l, self.n] = 0
                    path.engine._update_bits(l)
                    new_paths.append(path)
                else:
                    hard = 0 if current_llr >= 0 else 1
                    for bit in (0, 1):
                        new_path = Path(self.N, self.frozen_bits, llr)
                        new_path.pm = path.pm + (0.0 if bit == hard else abs(current_llr))
                        new_path.engine.L = path.engine.L.copy()
                        new_path.engine.B = path.engine.B.copy()
                        new_path.engine.B[l, self.n] = bit
                        new_path.engine._update_bits(l)
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_pass = []
        for path in paths:
            if self.crc_length > 0:
                u_hat = path.engine.B[:, self.n].astype(int)
                info_bits = u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(path)

        best = min(crc_pass, key=lambda p: p.pm) if crc_pass else min(paths, key=lambda p: p.pm)
        return best.engine.B[:, self.n].astype(int), best.pm
