"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _subtree_up(u_partial):
    """子树重编码部分和"""
    u = np.asarray(u_partial, dtype=int).copy()
    if len(u) == 1:
        return u
    half = len(u) // 2
    up_left = _subtree_up(u[:half])
    up_right = _subtree_up(u[half:])
    return np.concatenate([(up_left ^ up_right).astype(int), up_right])


def _bit_llr_at_phi(llr, u_hat, left, length, phi):
    """计算第 phi 个比特的 LLR"""
    if length == 1:
        return llr[0]
    half = length // 2
    if phi < left + half:
        ll = f_operation(llr[:half], llr[half:])
        return _bit_llr_at_phi(ll, u_hat, left, half, phi)
    up_left = _subtree_up(u_hat[left:left + half])
    lr = g_operation(llr[:half], llr[half:], up_left)
    return _bit_llr_at_phi(lr, u_hat, left + half, half, phi)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{"pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_val = _bit_llr_at_phi(llr_ch, path["u_hat"], 0, self.N, phi)

                if self.frozen_bits[phi]:
                    new_path = {"pm": path["pm"], "u_hat": path["u_hat"].copy()}
                    penalty = abs(llr_val) if llr_val < 0 else 0.0
                    new_path["pm"] += penalty
                    new_path["u_hat"][phi] = 0
                    new_paths.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = {"pm": path["pm"], "u_hat": path["u_hat"].copy()}
                        expected = 0 if llr_val >= 0 else 1
                        if u_bit != expected:
                            new_path["pm"] += abs(llr_val)
                        new_path["u_hat"][phi] = u_bit
                        new_paths.append(new_path)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if crc_pass:
                paths = crc_pass

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
