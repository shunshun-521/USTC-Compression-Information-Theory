"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _f_boxplus, _prepare_llr, g_operation, sc_decode


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(
        bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    )


def _pm_penalty(llr, u_bit):
    llr = np.clip(llr, -30.0, 30.0)
    return float(np.log1p(np.exp(-(1 - 2 * u_bit) * llr)))


def _list_decode_node(llr_node, frozen_node, list_size):
    """递归 SCL 列表译码"""
    n = len(llr_node)
    if n == 1:
        if frozen_node[0]:
            return [(np.array([0], dtype=int), np.array([0.0]), 0.0)]
        llr = llr_node[0]
        paths = [
            (np.array([0], dtype=int), np.array([0.0]), _pm_penalty(llr, 0)),
            (np.array([1], dtype=int), np.array([1.0]), _pm_penalty(llr, 1)),
        ]
        return sorted(paths, key=lambda x: x[2])[:list_size]

    half = n // 2
    llr_left = llr_node[:half]
    llr_right = llr_node[half:]
    llr_f = _f_boxplus(llr_left, llr_right)

    left_paths = _list_decode_node(llr_f, frozen_node[:half], list_size)

    all_paths = []
    for u_left, u_left_up, pm_left in left_paths:
        llr_g = g_operation(llr_left, llr_right, u_left_up)
        right_paths = _list_decode_node(llr_g, frozen_node[half:], list_size)
        for u_right, u_right_up, pm_right in right_paths:
            u_hat = np.concatenate([u_left, u_right])
            u_up = np.concatenate(
                [
                    np.bitwise_xor(u_left_up.astype(int), u_right_up.astype(int)).astype(
                        float
                    ),
                    u_right_up,
                ]
            )
            all_paths.append((u_hat, u_up, pm_left + pm_right))

    all_paths.sort(key=lambda x: x[2])
    return [(u, up, pm) for u, up, pm in all_paths[:list_size]]


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
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr = _prepare_llr(llr_ch)
        paths = _list_decode_node(llr, self.frozen_bits, self.list_size)

        if self.crc_length > 0:
            valid = []
            for u_hat, _, pm in paths:
                info = u_hat[self.info_indices]
                if crc_check(info, self.crc_length):
                    valid.append((u_hat, pm))
            if valid:
                u_hat, pm = min(valid, key=lambda x: x[1])
            else:
                u_hat, _, pm = paths[0]
        else:
            u_hat, _, pm = paths[0]

        return u_hat.copy(), pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    mismatches = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"L=1 SCL vs SC mismatches: {mismatches}/30")

    u_scl8, _ = SCLDecoder(N, frozen_bits, list_size=8).decode(llr)
    print("SCL L=8 ok, u_hat shape", u_scl8.shape)
