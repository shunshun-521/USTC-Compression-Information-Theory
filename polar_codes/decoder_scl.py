"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive, _xor_combine

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


def _path_metric(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器（基于树形 SC 的多路径扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数。"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for bit_idx in range(self.N):
            new_paths = []
            for path in paths:
                llr_val = self._get_bit_llr(path, bit_idx)
                if bit_idx in self.frozen_set:
                    self._set_bit(path, bit_idx, 0)
                    path['pm'] += _path_metric(llr_val, 0)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = self._clone_path(path)
                        self._set_bit(child, bit_idx, u_bit)
                        child['pm'] += _path_metric(llr_val, u_bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: (p['pm'], -sum(p['u'].values())))
            paths = new_paths[: self.list_size]

        best_crc = None
        best_crc_pm = float('inf')
        best = None
        best_pm = float('inf')

        for path in paths:
            u_hat = self._path_to_uhat(path)
            if path['pm'] < best_pm:
                best_pm = path['pm']
                best = u_hat
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length) and path['pm'] < best_crc_pm:
                    best_crc_pm = path['pm']
                    best_crc = u_hat

        result = best_crc if best_crc is not None else best
        return result, (best_crc_pm if best_crc is not None else best_pm)

    def _new_path(self, llr):
        return {'llr': llr, 'u': {}, 'left_arr': {}, 'pm': 0.0}

    def _clone_path(self, path):
        return {
            'llr': path['llr'],
            'u': path['u'].copy(),
            'left_arr': {k: v.copy() for k, v in path['left_arr'].items()},
            'pm': path['pm'],
        }

    def _path_to_uhat(self, path):
        u_hat = np.zeros(self.N, dtype=int)
        for k, v in path['u'].items():
            u_hat[k] = v
        return u_hat

    def _set_bit(self, path, bit_idx, val):
        path['u'][bit_idx] = val

    def _get_bit_llr(self, path, bit_idx):
        """沿树形结构获取 bit_idx 处的叶 LLR。"""
        return self._decode_subtree(path, path['llr'], 0, 0, bit_idx)

    def _decode_subtree(self, path, y, depth, node, target):
        if depth == self.n:
            return y[0] if node == target else 0.0

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]

        if target < node + half:
            left = f_operation(L1, L2)
            return self._decode_subtree(path, left, depth + 1, node, target)

        key = (depth, node)
        if key not in path['left_arr']:
            left_llr = f_operation(L1, L2)
            arr1 = self._decode_left(path, left_llr, depth + 1, node, half)
            path['left_arr'][key] = arr1

        arr1 = path['left_arr'][key]
        right = g_operation(L1, L2, arr1)
        return self._decode_subtree(path, right, depth + 1, node + half, target)

    def _decode_left(self, path, y, depth, node, half):
        """译码左子树，返回 arr1（与 SC 树形译码器一致）。"""
        if depth == self.n:
            bit_idx = node
            if bit_idx in path['u']:
                val = path['u'][bit_idx]
            elif bit_idx in self.frozen_set:
                val = 0
            else:
                val = 0 if y[0] >= 0 else 1
                path['u'][bit_idx] = val
            return [val]

        h = len(y) // 2
        L1, L2 = y[:h], y[h:]
        left_llr = f_operation(L1, L2)
        arr1 = self._decode_left(path, left_llr, depth + 1, node, h)
        right_llr = g_operation(L1, L2, arr1)
        arr2 = self._decode_left(path, right_llr, depth + 1, node + h, h)
        return _xor_combine(arr1, arr2)
