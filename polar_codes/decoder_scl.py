"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = _crc_poly(crc_length)
    reg = 0
    width = 8 if crc_length == 8 else 16
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(width):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length))


def _forced_decode_block(llr_node, frozen_node, offset, u_prefix, phi):
    """在已知前缀比特时译码子块，返回 u_hat 与 u_hat_up。"""
    n = len(llr_node)
    if n == 1:
        idx = offset
        if idx < phi:
            bit = int(u_prefix[idx])
        elif frozen_node[0]:
            bit = 0
        else:
            bit = 0 if llr_node[0] >= 0 else 1
        return np.array([bit], dtype=int), np.array([bit], dtype=int)

    half = n // 2
    llr_left = f_operation(llr_node[:half], llr_node[half:])
    u_left, u_left_up = _forced_decode_block(
        llr_left, frozen_node[:half], offset, u_prefix, phi
    )
    llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
    u_right, u_right_up = _forced_decode_block(
        llr_right, frozen_node[half:], offset + half, u_prefix, phi
    )
    u_hat = np.concatenate([u_left, u_right])
    u_hat_up = np.concatenate([u_left_up ^ u_right_up, u_right_up])
    return u_hat, u_hat_up


def _bit_llr(llr, u_prefix, phi, frozen_bits):
    """计算第 phi 位的 LLR。"""

    def decode_node(llr_node, frozen_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            if idx == phi:
                return None, None, llr_node[0]
            return None, None, None

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        if phi < offset + half:
            return decode_node(llr_left, frozen_node[:half], offset)

        u_left, u_left_up = _forced_decode_block(
            llr_left, frozen_node[:half], offset, u_prefix, phi
        )
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left_up)
        return decode_node(llr_right, frozen_node[half:], offset + half)

    _, _, llr_phi = decode_node(llr, frozen_bits, 0)
    return llr_phi


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        assert 2**self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        br = bit_reversal_permutation(self.N)
        llr = np.asarray(llr_ch, dtype=np.float64)[br]

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = _bit_llr(llr, path["u"], phi, self.frozen_bits)
                if self.frozen_bits[phi]:
                    bit = 0
                    hard = 0 if llr_phi >= 0 else 1
                    penalty = 0.0 if hard == 0 else abs(llr_phi)
                    new_u = path["u"].copy()
                    new_u[phi] = bit
                    candidates.append({"pm": path["pm"] + penalty, "u": new_u})
                else:
                    for bit in (0, 1):
                        hard = 0 if llr_phi >= 0 else 1
                        penalty = 0.0 if bit == hard else abs(llr_phi)
                        new_u = path["u"].copy()
                        new_u[phi] = bit
                        candidates.append(
                            {"pm": path["pm"] + penalty, "u": new_u}
                        )

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]


def scl_equivalent_sc(llr_ch, frozen_bits):
    """L=1 时 SCL 应等价于 SC。"""
    decoder = SCLDecoder(len(llr_ch), frozen_bits, list_size=1, crc_length=0)
    u_scl, _ = decoder.decode(llr_ch)
    u_sc = sc_decode_recursive(llr_ch, frozen_bits)
    return np.array_equal(u_scl, u_sc)
