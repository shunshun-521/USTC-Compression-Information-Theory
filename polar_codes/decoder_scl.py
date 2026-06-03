"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    register = 0
    for bit in info_bits:
        register <<= 1
        register |= int(bit)
        register &= (1 << crc_length) - 1
        if register & (1 << (crc_length - 1)):
            register ^= poly
            register &= (1 << crc_length) - 1

    for _ in range(crc_length):
        register <<= 1
        register &= (1 << crc_length) - 1
        if register & (1 << (crc_length - 1)):
            register ^= poly
            register &= (1 << crc_length) - 1

    crc_bits = np.array(
        [(register >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length <= 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


def _path_metric_update(pm, llr, u):
    expected = 0 if llr >= 0 else 1
    if u != expected:
        pm += abs(llr)
    return pm


def _butterfly_up(u_part):
    """蝶形重编码得到 up 向量"""
    u = np.array(u_part, dtype=int).copy()
    n = len(u)
    step = n // 2
    while step >= 1:
        for i in range(0, n, 2 * step):
            for j in range(i, i + step):
                u[j] ^= u[j + step]
        step //= 2
    return u


def _compute_llr_at_phi(llr, u, phi, frozen_bits):
    """计算第 phi 个比特的叶子 LLR（基于已译码 u[0:phi]）"""
    n = int(math.log2(len(llr)))

    def recurse(llr_node, frozen_node, bit_off):
        sz = len(llr_node)
        if sz == 1:
            return llr_node[0]
        half = sz // 2
        if phi < bit_off + half:
            llr_left = f_operation(llr_node[:half], llr_node[half:])
            return recurse(llr_left, frozen_node[:half], bit_off)
        u_left = u[bit_off : bit_off + half]
        u_left_up = _butterfly_up(u_left)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        return recurse(llr_right, frozen_node[half:], bit_off + half)

    return recurse(llr, frozen_bits, 0)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        if 2 ** self.n != N:
            raise ValueError(f"N={N} must be a power of 2")
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_leaf = _compute_llr_at_phi(llr, path["u"], phi, self.frozen_bits)
                if self.frozen_bits[phi]:
                    pm = _path_metric_update(path["pm"], llr_leaf, 0)
                    u = path["u"].copy()
                    u[phi] = 0
                    new_paths.append({"pm": pm, "u": u})
                else:
                    for bit in (0, 1):
                        pm = _path_metric_update(path["pm"], llr_leaf, bit)
                        u = path["u"].copy()
                        u[phi] = bit
                        new_paths.append({"pm": pm, "u": u})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            if valid:
                valid.sort(key=lambda p: p["pm"])
                best = valid[0]

        return best["u"], best["pm"]
