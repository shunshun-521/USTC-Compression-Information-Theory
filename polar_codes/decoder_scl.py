"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation, g_operation, prepare_channel_llr, sc_decode_recursive,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（自洽线性校验）"""
    info_bits = np.asarray(info_bits, dtype=int)
    n = len(info_bits)
    crc_bits = np.zeros(crc_length, dtype=int)
    for j in range(crc_length):
        val = 0
        for i, b in enumerate(info_bits):
            if (i + j + 1) % crc_length == 0:
                val ^= int(b)
        crc_bits[j] = val
    # 增强扩散：加入多项式反馈
    reg = 0
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    for i, b in enumerate(info_bits):
        reg ^= int(b) << (i % crc_length)
    for j in range(crc_length):
        crc_bits[j] ^= (reg >> j) & 1
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


def _get_bit_llr(llr_ch, frozen_bits, u_prefix, phi):
    """计算第 phi 个比特的 LLR，已知前 phi 个比特为 u_prefix"""
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_to_bit(llr_vec, frozen_vec, target, offset, u_known):
        n = len(llr_vec)
        if n == 1:
            return llr_vec[0]
        half = n // 2
        if target < offset + half:
            llr_left = f_operation(llr_vec[:half], llr_vec[half:])
            return decode_to_bit(llr_left, frozen_vec[:half], target, offset, u_known)
        else:
            llr_left = f_operation(llr_vec[:half], llr_vec[half:])
            u_left = u_known[offset:offset + half]
            # 重编码部分和
            u_left_up = u_left.copy()
            llr_right = g_operation(llr_vec[:half], llr_vec[half:], u_left_up)
            return decode_to_bit(
                llr_right, frozen_vec[half:], target, offset + half, u_known
            )

    return decode_to_bit(llr_ch, frozen_bits, phi, 0, u_prefix)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = prepare_channel_llr(llr_ch)
        paths = [(np.zeros(self.N, dtype=int), 0.0)]

        for phi in range(self.N):
            new_paths = []
            for u_hat, pm in paths:
                llr_val = _get_bit_llr(llr, self.frozen_bits, u_hat, phi)

                if self.frozen_bits[phi]:
                    u_hat_new = u_hat.copy()
                    u_hat_new[phi] = 0
                    new_paths.append((u_hat_new, pm + _pm_penalty(llr_val, 0)))
                else:
                    for u in (0, 1):
                        u_hat_new = u_hat.copy()
                        u_hat_new[phi] = u
                        new_paths.append((u_hat_new, pm + _pm_penalty(llr_val, u)))

            new_paths.sort(key=lambda x: x[1])
            paths = new_paths[:self.list_size]

        best_u, best_pm = min(paths, key=lambda x: x[1])

        if self.crc_length > 0:
            valid = [
                (u, pm) for u, pm in paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                best_u, best_pm = min(valid, key=lambda x: x[1])

        return best_u, best_pm

    def _crc_pass(self, u_hat):
        return crc_check(u_hat[self.info_indices], self.crc_length)
