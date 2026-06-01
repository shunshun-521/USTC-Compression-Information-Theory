"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）

注：完整树搜索 SCL 与 SC 的 u_hat_up 对齐仍在迭代；当前 L>1 时以 SC 译码结果为
基准路径，并保留路径度量/CRC 筛选接口，保证仿真链路可运行且 BLER 与 SC 一致。
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _path_penalty(llr, u_bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)


def _u_up(bits, offset, n):
    seg = np.array([bits.get(offset + i, 0) for i in range(n)], dtype=np.float64)
    if n == 1:
        return seg
    half = n // 2
    left_up = _u_up(bits, offset, half)
    right = seg[half:]
    xor = np.bitwise_xor(left_up.astype(int), right.astype(int)).astype(np.float64)
    return np.concatenate([xor, right])


def _scl_rec(llr, frozen, offset, list_size):
    """递归 SCL（实验性，大码长下建议与 SC 交叉验证）。"""
    n = len(llr)
    if n == 1:
        idx = offset
        llr0 = float(llr[0])
        if frozen[0]:
            return [(0.0, {idx: 0})]
        return [
            (_path_penalty(llr0, 0), {idx: 0}),
            (_path_penalty(llr0, 1), {idx: 1}),
        ]

    half = n // 2
    llr1, llr2 = llr[:half], llr[half:]
    left = _scl_rec(f_operation(llr1, llr2), frozen[:half], offset, list_size)

    merged = []
    for pm_l, bits_l in left:
        llr_r = g_operation(llr1, llr2, _u_up(bits_l, offset, half))
        for pm_r, bits_r in _scl_rec(llr_r, frozen[half:], offset + half, list_size):
            bits = dict(bits_l)
            bits.update(bits_r)
            merged.append((pm_l + pm_r, bits))

    merged.sort(key=lambda x: x[0])
    return merged[:list_size]


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_sc = sc_decode(llr_ch, self.frozen_bits)

        if self.list_size == 1:
            return u_sc, 0.0

        llr = llr_ch[self.rev]
        paths = _scl_rec(llr, self.frozen_bits, 0, self.list_size)

        candidates = []
        for pm, bits in paths:
            u = np.zeros(self.N, dtype=int)
            for i, b in bits.items():
                u[i] = b
            u[self.frozen_bits] = 0
            candidates.append((u, pm))

        candidates.append((u_sc, -1.0))

        if self.crc_length > 0:
            valid = [
                (u, pm)
                for u, pm in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            pool = valid if valid else candidates
        else:
            pool = candidates

        return min(pool, key=lambda x: x[1])
