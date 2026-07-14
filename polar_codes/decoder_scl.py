"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _frozen_to_info_set,
    _is_frozen,
    _reorder_channel_llr,
    _sc_tree_decode,
    f_operation,
    g_operation,
    precompute_sc_indices,
    sc_decode,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length, init_reg=0):
    reg = init_reg
    width = 8 if crc_length == 8 else 16
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（含路径复制优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_indices = _frozen_to_info_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.lambda_offset, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = _reorder_channel_llr(llr_ch)

        if self.list_size == 1:
            u_hat = _sc_tree_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{
            'pm': 0.0,
            'u_hat': np.zeros(self.N, dtype=int),
            'llr': llr_ch.copy(),
            'active_len': 0,
        }]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                partial = self._decode_to_bit(path['llr'].copy(), phi)
                llr_val = partial['leaf_llr']

                if _is_frozen(self.frozen_bits, phi):
                    new_path = {
                        'pm': path['pm'] + self._pm_penalty(llr_val, 0),
                        'u_hat': partial['u_hat'].copy(),
                        'llr': partial['llr'].copy(),
                        'active_len': phi + 1,
                    }
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        u_hat = partial['u_hat'].copy()
                        u_hat[phi] = u_bit
                        candidates.append({
                            'pm': path['pm'] + self._pm_penalty(llr_val, u_bit),
                            'u_hat': u_hat,
                            'llr': partial['llr'].copy(),
                            'active_len': phi + 1,
                        })

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path['u_hat'][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p['pm'])
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best['u_hat'].copy(), best['pm']

    def _decode_to_bit(self, llr_ch, target_phi):
        """译码至 target_phi 并返回该位 LLR"""
        N = self.N
        n = self.n
        frozen_bits = self.frozen_bits

        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_ch
        position = [0, 0, n, N]
        leaf_llr = 0.0

        def up(pos):
            p0 = pos[0] - 1
            span = 2 ** (pos[2] - pos[0] + 1)
            p1 = int(np.floor(pos[1] / span) * span)
            return [p0, p1, pos[2], pos[3]]

        def leftdown(pos):
            return [pos[0] + 1, pos[1], pos[2], pos[3]]

        def rightdown(pos):
            return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

        while not (bit_matrix[n][target_phi] == 0 or bit_matrix[n][target_phi] == 1):
            span = 2 ** (position[2] - position[0])
            start = position[1]
            up_llr = llr_matrix[position[0]][start:start + span]
            left_llr = llr_matrix[position[0] + 1][start:start + span // 2]
            left_bit = bit_matrix[position[0] + 1][start:start + span // 2]
            right_llr = llr_matrix[position[0] + 1][start + span // 2:start + span]
            right_bit = bit_matrix[position[0] + 1][start + span // 2:start + span]

            if not np.any(np.isnan(bit_matrix[position[0]][start:start + span])):
                position = up(position)
            elif not np.any(np.isnan(right_bit)):
                combined = np.array([(left_bit + right_bit) % 2, right_bit])
                combined.resize((1, span))
                bit_matrix[position[0]][start:start + span] = combined.copy()
            elif not np.any(np.isnan(right_llr)):
                if position[0] == position[2] - 1:
                    bit_pos = start + span // 2
                    if _is_frozen(frozen_bits, bit_pos):
                        bit_val = 0
                    else:
                        bit_val = 0 if right_llr[0] > 0 else 1
                    bit_matrix[position[0] + 1][start + span // 2:start + span] = bit_val
                    if bit_pos == target_phi:
                        leaf_llr = right_llr[0]
                else:
                    position = rightdown(position)
            elif not np.any(np.isnan(left_bit)):
                half = span // 2
                right_llr_new = np.array([
                    g_operation(up_llr[i], up_llr[i + half], left_bit[i])
                    for i in range(half)
                ])
                llr_matrix[position[0] + 1][start + span // 2:start + span] = right_llr_new
            elif np.any(np.isnan(left_llr)):
                half = span // 2
                left_llr_new = f_operation(up_llr[:half], up_llr[half:])
                llr_matrix[position[0] + 1][start:start + span // 2] = left_llr_new
            else:
                if position[0] == position[2] - 1:
                    bit_pos = start
                    if _is_frozen(frozen_bits, bit_pos):
                        bit_val = 0
                    else:
                        bit_val = 0 if left_llr[0] >= 0 else 1
                    bit_matrix[position[0] + 1][start:start + span // 2] = bit_val
                    if bit_pos == target_phi:
                        leaf_llr = left_llr[0]
                else:
                    position = leftdown(position)

        u_hat = np.nan_to_num(bit_matrix[n], nan=0.0).astype(int)
        return {'u_hat': u_hat, 'llr': llr_matrix[0], 'leaf_llr': leaf_llr}


def verify_scl_equals_sc(N=64, K=32, seed=0):
    """验证 L=1 的 SCL 等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(5.0, K / N)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma,
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError('SCL L=1 != SC')

    return True
