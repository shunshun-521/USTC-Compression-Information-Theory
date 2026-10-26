"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("Unsupported CRC length")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    reg = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int64
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int64)
    return _crc_remainder(bits, crc_length) == 0


def _bit_llr(llr_ch, u_hat, phi):
    """给定前缀 u_hat[0:phi]，计算比特 phi 的 LLR。"""

    def rec(llr_node, bit_offset, n_stage):
        if n_stage == 0:
            return llr_node[0]
        half = len(llr_node) // 2
        if phi < bit_offset + half:
            return rec(
                f_operation(llr_node[:half], llr_node[half:]),
                bit_offset,
                n_stage - 1,
            )
        u_left = u_hat[bit_offset: bit_offset + half]
        return rec(
            g_operation(llr_node[:half], llr_node[half:], u_left),
            bit_offset + half,
            n_stage - 1,
        )

    n = int(math.log2(len(llr_ch)))
    return rec(np.asarray(llr_ch, dtype=np.float64), 0, n)


class _Path:
    __slots__ = ("pm", "u_hat", "llr_ch")

    def __init__(self, N, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int64)
        self.llr_ch = llr_ch


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, llr_ch)]

        for phi in range(self.N):
            candidates = []
            if self.frozen_bits[phi]:
                for path in paths:
                    llr = _bit_llr(path.llr_ch, path.u_hat, phi)
                    new_pm = path.pm + (abs(llr) if llr < 0 else 0.0)
                    candidates.append((new_pm, path, 0))
            else:
                for path in paths:
                    llr = _bit_llr(path.llr_ch, path.u_hat, phi)
                    for bit in (0, 1):
                        penalty = (
                            0.0
                            if (llr >= 0 and bit == 0) or (llr < 0 and bit == 1)
                            else abs(llr)
                        )
                        candidates.append((path.pm + penalty, path, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, bit in candidates[: self.L]:
                child = _Path(self.N, parent.llr_ch)
                child.pm = pm
                child.u_hat = parent.u_hat.copy()
                child.u_hat[phi] = bit
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        return best.u_hat.astype(int), best.pm
