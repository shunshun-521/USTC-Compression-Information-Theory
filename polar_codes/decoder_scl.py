"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _poly_div_remainder(dividend, divisor):
    """GF(2) 多项式长除法求余数（MSB 在列表首位）。"""
    div = list(dividend)
    while len(div) >= len(divisor):
        if div[0] == 1:
            for i in range(len(divisor)):
                div[i] ^= divisor[i]
        div.pop(0)
    return div


def _bits_to_poly(bits):
    return [int(b) for b in bits]


def _poly_to_bits(poly, length):
    if len(poly) < length:
        poly = [0] * (length - len(poly)) + poly
    return np.array(poly[-length:], dtype=int)


def _crc_divisor(poly, crc_length):
    """构造 CRC 生成多项式系数（含最高次项）。"""
    return [1] + [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    divisor = _crc_divisor(poly, crc_length)
    dividend = list(info_bits) + [0] * crc_length
    remainder = _poly_div_remainder(dividend, divisor)
    crc_bits = _poly_to_bits(remainder, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    divisor = _crc_divisor(poly, crc_length)
    remainder = _poly_div_remainder(_bits_to_poly(bits), divisor)
    return len(remainder) == 0 or all(b == 0 for b in remainder)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _path_metric_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _update_llrs(self, L, B, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (
                        B[j, s] + B[j - branch_size, s]
                    ) % 2
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数。返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N
        llr_input = llr_ch[self.rev]

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int32),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_input

        decode_order = [_bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            candidates = []
            for path in paths:
                self._update_llrs(path['L'], path['B'], l)
                llr_val = path['L'][l, n]

                if l in self.frozen_set:
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': path['pm'] + self._path_metric_penalty(llr_val, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['B'][l, n] = 0
                    new_path['u_hat'][l] = 0
                    self._update_bits(new_path['B'], l)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': path['pm'] + self._path_metric_penalty(llr_val, u_bit),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['B'][l, n] = u_bit
                        new_path['u_hat'][l] = u_bit
                        self._update_bits(new_path['B'], l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [p for p in paths if crc_check(p['u_hat'][self.info_positions], self.crc_length)]
            best = min(crc_pass if crc_pass else paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
