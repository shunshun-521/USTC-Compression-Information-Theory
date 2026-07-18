"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    f_operation,
    g_operation,
    precompute_sc_indices,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


# ==================== SCL 译码器 ====================


class PathState:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.phases, self.llr_layer_vec, self.bit_layer_vec, _ = (
            precompute_sc_indices(N)
        )

    @staticmethod
    def _branch_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _compute_llr(self, path, phase, phi):
        for s in self.llr_layer_vec[phi]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phase, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )
        return path.L[phase, self.n]

    def _update_bits(self, path, phase, phi, bit):
        path.u_hat[phase] = bit
        path.B[phase, self.n] = bit
        if phase < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(phase, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(phase, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        root = PathState(self.N, self.n)
        root.L[:, 0] = llr_ch
        paths = [root]

        for phi, phase in enumerate(self.phases):
            candidates = []
            for path in paths:
                llr = self._compute_llr(path, phase, phi)
                if phase in self.frozen_set:
                    pm = path.pm + self._branch_penalty(llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for bit in (0, 1):
                        pm = path.pm + self._branch_penalty(llr, bit)
                        candidates.append((pm, path, bit))

            candidates.sort(key=lambda item: item[0])
            survivors = candidates[: self.list_size]

            new_paths = []
            for pm, old_path, bit in survivors:
                new_path = PathState(self.N, self.n)
                new_path.pm = pm
                new_path.L = old_path.L.copy()
                new_path.B = old_path.B.copy()
                new_path.u_hat = old_path.u_hat.copy()
                self._update_bits(new_path, phase, phi, bit)
                new_paths.append(new_path)
            paths = new_paths

        best = min(paths, key=lambda p: p.pm)
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_positions], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm
