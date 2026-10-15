"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, _g_scalar

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    rem = _crc_remainder(padded, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


def _subtree_codeword(indices, u_hat):
    if len(indices) == 1:
        return [int(u_hat[indices[0]])]
    half = len(indices) // 2
    left = indices[:half]
    right = indices[half:]
    el = _subtree_codeword(left, u_hat)
    er = _subtree_codeword(right, u_hat)
    return [el[i] ^ er[i] for i in range(half)] + er


def _llr_at_phi(phi, llr, u_hat):
    """给定前缀 u_hat[0:phi] 已判决，计算比特 phi 的 LLR。"""

    def rec(llrs, indices):
        if len(indices) == 1:
            return llrs[0]
        half = len(indices) // 2
        left = indices[:half]
        right = indices[half:]
        if phi in left:
            new_llrs = [
                float(f_operation(llrs[j], llrs[j + half])) for j in range(half)
            ]
            return rec(new_llrs, left)
        enc_left = _subtree_codeword(left, u_hat)
        new_llrs = [
            _g_scalar(llrs[j], llrs[j + half], enc_left[j]) for j in range(half)
        ]
        return rec(new_llrs, right)

    return rec(np.asarray(llr, dtype=np.float64).tolist(), list(range(len(llr))))


class SCLDecoder:
    """SCL 译码器（逐比特 LLR + 路径裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [(np.zeros(self.N, dtype=int), 0.0)]

        for phi in range(self.N):
            candidates = []
            for u_hat, pm in paths:
                llr_bit = _llr_at_phi(phi, llr_ch, u_hat)
                if self.frozen_bits[phi]:
                    bit = 0
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    u_new = u_hat.copy()
                    u_new[phi] = 0
                    candidates.append((u_new, pm + penalty))
                else:
                    for bit in (0, 1):
                        llr_sign_bit = 0 if llr_bit >= 0 else 1
                        penalty = 0.0 if bit == llr_sign_bit else abs(llr_bit)
                        u_new = u_hat.copy()
                        u_new[phi] = bit
                        candidates.append((u_new, pm + penalty))

            candidates.sort(key=lambda item: item[1])
            paths = candidates[: self.list_size]

        best_u, best_pm = min(paths, key=lambda item: item[1])
        if self.crc_length > 0:
            valid = [
                (u, pm)
                for u, pm in paths
                if crc_check(u[self.info_positions], self.crc_length)
            ]
            if valid:
                best_u, best_pm = min(valid, key=lambda item: item[1])

        return best_u.astype(int), best_pm
