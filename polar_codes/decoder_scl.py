"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import SCDecoderCore, active_bit_level, active_llr_level, f_operation, g_operation
from encoder import bit_reversal_permutation



def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return [1, 0, 0, 0, 0, 1, 1, 1, 1]
    if crc_length == 16:
        # x^16 + x^12 + x^5 + 1 (0x8005)
        poly = [0] * 17
        poly[0] = 1
        poly[4] = 1
        poly[11] = 1
        poly[16] = 1
        return poly
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def _crc_remainder(bits, crc_length):
    poly = _crc_poly_bits(crc_length)
    msg = list(map(int, bits))
    n = len(poly)
    for i in range(len(msg) - n + 1):
        if msg[i]:
            for j in range(n):
                msg[i + j] ^= poly[j]
    return np.array(msg[-(n - 1):], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(padded, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    remainder = _crc_remainder(bits, crc_length)
    return np.all(remainder == 0)


class PathState:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = None if info_indices is None else np.asarray(info_indices, dtype=int)
        self.leaf_order = bit_reversal_permutation(N)

    @staticmethod
    def _pm_update(pm, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        if u_bit != hard:
            pm += abs(llr_val)
        return pm

    def _update_llrs(self, path, leaf_idx):
        for stage in range(self.n - active_llr_level(leaf_idx, self.n), self.n):
            block_size = 1 << (stage + 1)
            branch_size = block_size >> 1
            for j in range(leaf_idx, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, stage + 1] = f_operation(
                        path.L[j, stage], path.L[j + branch_size, stage]
                    )
                else:
                    path.L[j, stage + 1] = g_operation(
                        path.L[j - branch_size, stage],
                        path.L[j, stage],
                        path.B[j - branch_size, stage + 1],
                    )

    def _update_bits(self, path, leaf_idx):
        if leaf_idx < self.N // 2:
            return
        for stage in range(self.n, self.n - active_bit_level(leaf_idx, self.n), -1):
            block_size = 1 << stage
            branch_size = block_size >> 1
            for j in range(leaf_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, stage - 1] = (
                        path.B[j, stage] ^ path.B[j - branch_size, stage]
                    )
                    path.B[j, stage - 1] = path.B[j, stage]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [PathState(self.N, self.n, llr_ch)]

        for leaf_idx in self.leaf_order:
            candidates = []
            for path in paths:
                self._update_llrs(path, leaf_idx)
                llr_val = path.L[leaf_idx, self.n]
                if self.frozen_bits[leaf_idx]:
                    candidates.append((self._pm_update(path.pm, llr_val, 0), path, 0))
                else:
                    for u_bit in (0, 1):
                        candidates.append((self._pm_update(path.pm, llr_val, u_bit), path, u_bit))

            candidates.sort(key=lambda item: item[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for new_pm, parent, u_bit in selected:
                child = copy.deepcopy(parent)
                child.pm = new_pm
                child.u_hat[leaf_idx] = u_bit
                child.B[leaf_idx, self.n] = u_bit
                self._update_bits(child, leaf_idx)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            def _crc_ok(path):
                bits = path.u_hat[self.info_indices] if self.info_indices is not None else path.u_hat
                return crc_check(bits, self.crc_length)

            valid = [p for p in paths if _crc_ok(p)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
