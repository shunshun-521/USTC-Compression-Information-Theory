"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _local_butterfly


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位计算并附加"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class _Path:
    __slots__ = ("u_hat", "pm", "active")

    def __init__(self, N):
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.pm = 0.0
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：按路径复制 u_hat 与度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _llr_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        paths = [_Path(self.N)]
        paths[0].pm = 0.0

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                llr_phi = self._compute_llr(path.u_hat, llr, phi)
                if self.frozen_bits[phi]:
                    path.pm += self._llr_penalty(llr_phi, 0)
                    path.u_hat[phi] = 0
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N)
                        child.u_hat[:] = path.u_hat
                        child.u_hat[phi] = bit
                        child.pm = path.pm + self._llr_penalty(llr_phi, bit)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = self._select_path(paths)
        return best.u_hat.astype(int), best.pm

    def _compute_llr(self, u_partial, llr, phi):
        """计算当前位 phi 的 LLR（递归 f/g，使用已判决前缀）"""
        return self._node_llr(llr, self.n, 0, phi, u_partial)

    def _node_llr(self, node_llr, depth, bit_index, target_phi, u_hat):
        if depth == 0:
            return node_llr[0]
        half = 1 << (depth - 1)
        left_llr = f_operation(node_llr[:half], node_llr[half:])
        if target_phi < bit_index + half:
            return self._node_llr(left_llr, depth - 1, bit_index, target_phi, u_hat)
        v_left = _local_butterfly(u_hat[bit_index : bit_index + half])
        right_llr = g_operation(node_llr[:half], node_llr[half:], v_left)
        return self._node_llr(
            right_llr, depth - 1, bit_index + half, target_phi, u_hat
        )

    def _select_path(self, paths):
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            passing = []
            for p in paths:
                info_bits = p.u_hat[info_idx]
                if crc_check(info_bits, self.crc_length):
                    passing.append(p)
            if passing:
                return min(passing, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
