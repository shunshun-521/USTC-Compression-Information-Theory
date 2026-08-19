"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation, g_operation, _compute_left_alpha, _compute_right_alpha,
    _compute_encoding_step, _position_state,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    else:
        poly = 0x8005

    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.mask = 1 - self.frozen_bits
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_llr_structure(self, received_llr):
        layers = [received_llr.copy()]
        length = self.N // 2
        while length > 0:
            layers.append(np.zeros(length, dtype=np.float64))
            length //= 2
        return layers

    def _copy_llr_layers(self, layers):
        return [layer.copy() for layer in layers]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size

        paths = [{
            'llr': self._init_llr_structure(llr_ch),
            'bits': [np.zeros(self.N, dtype=np.int8) for _ in range(self.n + 1)],
            'pm': 0.0,
            'prev_state': np.ones(self.n, dtype=np.int8),
        }]

        for position in range(self.N):
            candidates = []
            for path in paths:
                current_state = _position_state(position, self.n)
                llr_layers = path['llr']
                bits_layers = path['bits']

                for i in range(1, self.n + 1):
                    if current_state[i - 1] == path['prev_state'][i - 1]:
                        continue
                    llr = llr_layers[i - 1]
                    if current_state[i - 1] == 0:
                        llr_layers[i] = _compute_left_alpha(llr)
                    else:
                        end = position
                        start = end - (2 ** (self.n - i))
                        left_bits = bits_layers[i][start:end]
                        llr_layers[i] = _compute_right_alpha(llr, left_bits)

                cur_llr = llr_layers[self.n][0]

                if self.mask[position] == 0:
                    new_path = {
                        'llr': self._copy_llr_layers(llr_layers),
                        'bits': [b.copy() for b in bits_layers],
                        'pm': path['pm'] + (0.0 if cur_llr >= 0 else abs(cur_llr)),
                        'prev_state': current_state.copy(),
                    }
                    new_path['bits'][self.n][position] = 0
                    for i in range(self.n - 1, -1, -1):
                        new_path['bits'][i] = _compute_encoding_step(
                            i, self.n, new_path['bits'][i + 1], new_path['bits'][i]
                        )
                    candidates.append(new_path)
                else:
                    for bit_val in (0, 1):
                        penalty = 0.0 if (
                            (bit_val == 0 and cur_llr >= 0) or
                            (bit_val == 1 and cur_llr < 0)
                        ) else abs(cur_llr)
                        new_path = {
                            'llr': self._copy_llr_layers(llr_layers),
                            'bits': [b.copy() for b in bits_layers],
                            'pm': path['pm'] + penalty,
                            'prev_state': current_state.copy(),
                        }
                        new_path['bits'][self.n][position] = bit_val
                        for i in range(self.n - 1, -1, -1):
                            new_path['bits'][i] = _compute_encoding_step(
                                i, self.n, new_path['bits'][i + 1], new_path['bits'][i]
                            )
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:L]

        if self.crc_length > 0:
            info_idx = np.where(self.mask == 1)[0]
            valid = []
            for p in paths:
                u = p['bits'][self.n]
                if len(info_idx) >= self.crc_length:
                    check_bits = u[info_idx]
                    if crc_check(check_bits, self.crc_length):
                        valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p['pm'])
        return best['bits'][self.n].astype(int), best['pm']
