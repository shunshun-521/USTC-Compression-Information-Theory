"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive
from encoder import prepare_decoder_llr

CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]


def _poly_bits(crc_length):
    if crc_length == 8:
        return CRC8_POLY_BITS
    return [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_mod2_divide(message_bits, crc_length):
    g = _poly_bits(crc_length)
    msg = list(map(int, message_bits)) + [0] * crc_length
    for i in range(len(message_bits)):
        if msg[i]:
            for j in range(len(g)):
                if i + j < len(msg):
                    msg[i + j] ^= g[j]
    return np.array(msg[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits_arr = _crc_mod2_divide(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits_arr])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    g = _poly_bits(crc_length)
    msg = list(map(int, bits))
    for i in range(len(bits) - crc_length):
        if msg[i]:
            for j in range(len(g)):
                if i + j < len(msg):
                    msg[i + j] ^= g[j]
    return sum(msg[-crc_length:]) == 0


class SCLDecoder:
    """SCL 译码器（递归路径分裂）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, bit):
        hard = 1 if llr < 0 else 0
        return 0.0 if bit == hard else abs(llr)

    def _subtree_return(self, u_hat, node_idx, length):
        """左子树递归返回的部分码字（与 SC 一致）"""
        if length == 1:
            return np.array([u_hat[node_idx]])
        half = length // 2
        left_ret = self._subtree_return(u_hat, 2 * node_idx, half)
        right_ret = self._subtree_return(u_hat, 2 * node_idx + 1, half)
        return np.concatenate([(left_ret + right_ret) % 2, right_ret])

    def _scl_recursive(self, y, node_idx, paths):
        if len(y) == 1:
            new_paths = []
            for pm, u_hat in paths:
                if self.frozen_bits[node_idx]:
                    uh = u_hat.copy()
                    uh[node_idx] = 0
                    new_paths.append((pm + self._pm_penalty(y[0], 0), uh))
                else:
                    for bit in (0, 1):
                        uh = u_hat.copy()
                        uh[node_idx] = bit
                        new_paths.append(
                            (pm + self._pm_penalty(y[0], bit), uh)
                        )
            new_paths.sort(key=lambda x: x[0])
            return new_paths[: self.list_size]

        half = len(y) // 2
        left_llr = f_operation(y[:half], y[half:])
        left_paths = self._scl_recursive(
            left_llr, 2 * node_idx, [(pm, uh.copy()) for pm, uh in paths]
        )

        right_paths = []
        for pm, u_hat in left_paths:
            left_partial = self._subtree_return(u_hat, 2 * node_idx, half)
            right_llr = g_operation(y[:half], y[half:], left_partial)
            sub = self._scl_recursive(
                right_llr, 2 * node_idx + 1, [(pm, u_hat)]
            )
            right_paths.extend(sub)

        right_paths.sort(key=lambda x: x[0])
        return right_paths[: self.list_size]

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        llr_ch = prepare_decoder_llr(llr_ch)
        u0 = np.zeros(self.N, dtype=int)
        paths = self._scl_recursive(llr_ch, 0, [(0.0, u0)])

        if self.crc_length > 0:
            valid = [
                (pm, uh)
                for pm, uh in paths
                if crc_check(uh[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x[0])
        return best[1].copy(), best[0]
