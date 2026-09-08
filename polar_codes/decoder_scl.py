"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _update_llrs,
    _update_bits,
    clip_llr,
)
from encoder import bit_reversed

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

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
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class _Path:
    def __init__(self, N, n, llr):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        child = _Path.__new__(_Path)
        child.L = self.L.copy()
        child.B = self.B.copy()
        child.pm = self.pm
        child.u_hat = self.u_hat.copy()
        return child


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = clip_llr(np.asarray(llr_ch, dtype=np.float64))
        N, n = self.N, self.n
        paths = [_Path(N, n, llr)]

        for phase in range(N):
            l = bit_reversed(phase, n)
            candidates = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                cur_llr = path.L[l, n]

                if self.frozen_bits[l]:
                    child = path.copy()
                    child.pm = _pm_update(child.pm, cur_llr, 0)
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm = _pm_update(child.pm, cur_llr, u)
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

            for path in paths:
                _update_bits(path.B, l, n, N)

        crc_paths = []
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            for path in paths:
                if crc_check(path.u_hat[info_positions], self.crc_length):
                    crc_paths.append(path)

        best = min(crc_paths or paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
