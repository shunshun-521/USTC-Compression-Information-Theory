"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc_ops import f_operation, g_operation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def pack_bits_msb(bits):
    """将比特序列 MSB 优先打包为字节"""
    bits = list(np.asarray(bits, dtype=int))
    if not bits:
        return bytes()
    nbytes = (len(bits) + 7) // 8
    padded = bits + [0] * (nbytes * 8 - len(bits))
    out = bytearray()
    for i in range(nbytes):
        val = 0
        for j in range(8):
            val = (val << 1) | padded[i * 8 + j]
        out.append(val)
    return bytes(out)


def _crc8_bytes(data):
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ _CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_bytes(data):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    data = pack_bits_msb(info_bits)
    if crc_length == 8:
        crc_val = _crc8_bytes(data)
    else:
        crc_val = _crc16_bytes(data)
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    data_bits = bits[:-crc_length]
    expected = crc_encode(data_bits, crc_length)
    return np.array_equal(expected, bits)


def _all_decided(bits):
    return not np.any(np.isnan(bits))


def _pm_update(llr, bit):
    """路径度量增量：与 LLR 符号不一致时加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def sc_step_to_bit(llr_matrix, bit_matrix, frozen_bits, target_bit):
    """
    从当前矩阵状态出发，执行 SC 直至 target_bit 判决完成。
    返回更新后的 llr_matrix, bit_matrix, 以及 target_bit 处的 LLR。
    """
    N = llr_matrix.shape[1]
    n = int(math.log2(N))
    position = [0, 0, n, N]

    def up(pos):
        return [pos[0] - 1,
                int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1))),
                pos[2], pos[3]]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - pos[0] - 1), pos[2], pos[3]]

    target_llr = 0.0

    while math.isnan(bit_matrix[n][target_bit]):
        r, c, max_r, width = position
        span = 2 ** (max_r - r)
        up_llr = llr_matrix[r][c:c + span]
        up_bit = bit_matrix[r][c:c + span]
        half = span // 2
        left_llr = llr_matrix[r + 1][c:c + half]
        left_bit = bit_matrix[r + 1][c:c + half]
        right_llr = llr_matrix[r + 1][c + half:c + span]
        right_bit = bit_matrix[r + 1][c + half:c + span]

        if _all_decided(up_bit):
            position = up(position)
            continue

        if _all_decided(right_bit):
            combined = np.empty(span, dtype=np.float64)
            combined[:half] = (left_bit + right_bit) % 2
            combined[half:] = right_bit
            bit_matrix[r][c:c + span] = combined
            continue

        if _all_decided(right_llr):
            if r == max_r - 1:
                bit_pos = c + half
                if frozen_bits[bit_pos]:
                    val = 0.0
                else:
                    val = 0.0 if right_llr[0] >= 0 else 1.0
                bit_matrix[r + 1][c + half] = val
                if bit_pos == target_bit:
                    target_llr = right_llr[0]
            else:
                position = rightdown(position)
            continue

        if _all_decided(left_bit):
            right_llr_new = g_operation(
                up_llr[:half], up_llr[half:], left_bit.astype(int)
            )
            llr_matrix[r + 1][c + half:c + span] = right_llr_new
            continue

        if not _all_decided(left_llr):
            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[r + 1][c:c + half] = left_llr_new
            continue

        if r == max_r - 1:
            bit_pos = c
            if frozen_bits[bit_pos]:
                val = 0.0
            else:
                val = 0.0 if left_llr[0] >= 0 else 1.0
            bit_matrix[r + 1][c] = val
            if bit_pos == target_bit:
                target_llr = left_llr[0]
        else:
            position = leftdown(position)

    return llr_matrix, bit_matrix, target_llr


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _new_path(self, llr_ch):
        llr_matrix = np.full((self.n + 1, self.N), np.nan)
        bit_matrix = np.full((self.n + 1, self.N), np.nan)
        llr_matrix[0] = llr_ch.copy()
        return {
            'llr': llr_matrix,
            'bit': bit_matrix,
            'pm': 0.0,
        }

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            from decoder_sc import sc_decode_factor_graph
            u_hat = sc_decode_factor_graph(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            candidates = []

            for path in paths:
                llr_m, bit_m, bit_llr = sc_step_to_bit(
                    path['llr'].copy(), path['bit'].copy(),
                    self.frozen_bits, phi
                )

                if self.frozen_bits[phi]:
                    pm = path['pm'] + _pm_update(bit_llr, 0)
                    bit_m[self.n][phi] = 0
                    candidates.append({
                        'llr': llr_m, 'bit': bit_m, 'pm': pm,
                    })
                else:
                    for bit_val in (0, 1):
                        bm = bit_m.copy()
                        bm[self.n][phi] = bit_val
                        pm = path['pm'] + _pm_update(bit_llr, bit_val)
                        candidates.append({
                            'llr': llr_m.copy(), 'bit': bm, 'pm': pm,
                        })

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:self.list_size]

        best_path = None
        if self.crc_length > 0:
            for path in sorted(paths, key=lambda p: p['pm']):
                u_hat = path['bit'][self.n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    best_path = path
                    break

        if best_path is None:
            best_path = min(paths, key=lambda p: p['pm'])

        u_hat = best_path['bit'][self.n].astype(int)
        return u_hat, best_path['pm']
