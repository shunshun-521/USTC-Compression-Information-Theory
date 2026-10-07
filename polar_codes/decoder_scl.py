"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _align_channel_llr

# ==================== CRC 工具 ====================

_CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]  # x^8 + x^2 + x + 1
_CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]  # CRC-16-IBM


def _crc_generator(crc_length):
    if crc_length == 8:
        return _CRC8_GEN
    if crc_length == 16:
        return _CRC16_GEN
    raise ValueError("crc_length must be 8 or 16")


def _gf2_crc_remainder(bits, gen):
    r = len(gen) - 1
    msg = [int(b) for b in bits]
    n_info = len(msg) - r
    for i in range(n_info):
        if msg[i]:
            for j, g in enumerate(gen):
                if g:
                    msg[i + j] ^= 1
    return np.array(msg[n_info : n_info + r], dtype=np.int8)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    gen = _crc_generator(crc_length)
    r = crc_length
    augmented = np.concatenate([info_bits, np.zeros(r, dtype=np.int8)])
    crc_bits = _gf2_crc_remainder(augmented, gen)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    gen = _crc_generator(crc_length)
    r = crc_length
    msg = [int(b) for b in bits]
    for i in range(len(msg) - r):
        if msg[i]:
            for j, g in enumerate(gen):
                if g:
                    msg[i + j] ^= 1
    return sum(msg[-r:]) == 0


def _llr_at_bit(llr_ch, u_hat, phi):
    """给定前缀判决 u_hat[0:phi]，计算比特 phi 的 LLR。"""

    def recurse(seg, offset, length):
        if length == 1:
            return float(seg[0])
        half = length // 2
        left = f_operation(seg[:half], seg[half:])
        if phi < offset + half:
            return recurse(left, offset, half)
        u_left = u_hat[offset : offset + half]
        right = g_operation(seg[:half], seg[half:], u_left)
        return recurse(right, offset + half, half)

    return recurse(np.asarray(llr_ch, dtype=np.float64), 0, len(llr_ch))


def _pm_penalty(llr_bit, u_bit):
    hard = 0 if llr_bit >= 0 else 1
    return 0.0 if int(u_bit) == hard else abs(llr_bit)


class SCLDecoder:
    """SCL 译码器（路径列表 + CRC 辅助）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        assert 2 ** self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        llr_ch = _align_channel_llr(llr_ch)
        N = self.N

        if self.list_size == 1:
            from decoder_sc import sc_decode

            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [(0.0, np.zeros(N, dtype=np.int8))]

        for phi in range(N):
            new_paths = []
            for pm, u_hat in paths:
                llr_bit = _llr_at_bit(llr_ch, u_hat, phi)
                if self.frozen_bits[phi]:
                    u = 0
                    u_new = u_hat.copy()
                    u_new[phi] = u
                    new_paths.append((pm + _pm_penalty(llr_bit, u), u_new))
                else:
                    for u in (0, 1):
                        u_new = u_hat.copy()
                        u_new[phi] = u
                        new_paths.append((pm + _pm_penalty(llr_bit, u), u_new))
            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        best_pm, best_u = paths[0]
        if self.crc_length > 0:
            passed = [
                (pm, u)
                for pm, u in paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if passed:
                best_pm, best_u = min(passed, key=lambda x: x[0])

        return best_u.copy(), best_pm
