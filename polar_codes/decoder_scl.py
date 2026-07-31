"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    register = 0
    for bit in info_bits:
        register = ((register << 1) | int(bit)) & ((1 << crc_length) - 1)
        if register & (1 << (crc_length - 1)):
            register ^= poly
    for _ in range(crc_length):
        register = (register << 1) & ((1 << crc_length) - 1)
        if register & (1 << (crc_length - 1)):
            register ^= poly
    crc_bits = np.array(
        [(register >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    poly = CRC_POLYNOMIALS[crc_length]
    bits = np.asarray(bits, dtype=int)
    register = 0
    for bit in bits:
        register = ((register << 1) | int(bit)) & ((1 << crc_length) - 1)
        if register & (1 << (crc_length - 1)):
            register ^= poly
    return register == 0


def _partial_s(lo, hi, u_hat):
    """根据已知 u_hat[lo:hi] 计算 aff3ct 风格部分和 s。"""
    n = hi - lo
    if n == 1:
        return np.array([u_hat[lo]], dtype=int)
    mid = lo + n // 2
    s_left = _partial_s(lo, mid, u_hat)
    s_right = _partial_s(mid, hi, u_hat)
    s = np.zeros(n, dtype=int)
    s[: n // 2] = s_left ^ s_right
    s[n // 2 :] = s_right
    return s


def _compute_llr_at_bit(llr_ch, u_prefix, phi):
    """给定前缀 u_prefix，计算第 phi 位的 LLR。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)

    def recurse(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            return llr_node[0]
        half = n // 2
        mid = bit_offset + half
        if phi < mid:
            llr_left = f_operation(llr_node[:half], llr_node[half:])
            return recurse(llr_left, bit_offset)
        s_left = _partial_s(bit_offset, mid, u_prefix)
        llr_right = g_operation(llr_node[:half], llr_node[half:], s_left)
        return recurse(llr_right, mid)

    return recurse(llr_ch, 0)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [{'u': np.zeros(self.N, dtype=int), 'pm': 0.0}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = _compute_llr_at_bit(llr_ch, path['u'], phi)
                if self.frozen_bits[phi]:
                    u_new = path['u'].copy()
                    u_new[phi] = 0
                    pm_new = path['pm'] + self._pm_penalty(llr_phi, 0)
                    candidates.append({'u': u_new, 'pm': pm_new})
                else:
                    for u_bit in (0, 1):
                        u_new = path['u'].copy()
                        u_new[phi] = u_bit
                        pm_new = path['pm'] + self._pm_penalty(llr_phi, u_bit)
                        candidates.append({'u': u_new, 'pm': pm_new})

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        crc_valid = []
        if self.crc_length > 0:
            for path in paths:
                info_bits = path['u'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)

        pool = crc_valid if crc_valid else paths
        best = min(pool, key=lambda p: p['pm'])
        return best['u'], best['pm']
