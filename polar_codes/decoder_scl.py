"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode, precompute_sc_indices, _build_polar_tree, _allocate_lambdas, _init_frozen, _decode_tree, _collect_bits, f_operation, g_operation, _h_decision


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(expected, bits)


def sc_llr_for_bit(llr, u_hat, phi, N):
    """计算 SC 译码第 phi 位时的 LLR。"""
    n = int(np.log2(N))

    def recurse(llr_node, depth, bit_start):
        if depth == n:
            return llr_node[0]
        half = 1 << (n - depth - 1)
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        if phi < bit_start + half:
            return recurse(llr_left, depth + 1, bit_start)
        u_left = u_hat[bit_start:bit_start + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        return recurse(llr_right, depth + 1, bit_start + half)

    return recurse(np.asarray(llr, dtype=np.float64).copy(), 0, 0)


class Path:
    """单条 SCL 路径。"""

    __slots__ = ('llr', 'u_hat', 'pm')

    def __init__(self, llr, N):
        self.llr = llr
        self.u_hat = np.zeros(N, dtype=int)
        self.pm = 0.0

    def copy(self):
        p = Path(self.llr.copy(), len(self.u_hat))
        p.u_hat = self.u_hat.copy()
        p.pm = self.pm
        return p


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]
        precompute_sc_indices(N)

    def _path_metric_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr, self.frozen_bits)
            return u_hat, 0.0

        N = self.N
        paths = [Path(llr.copy(), N)]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr_phi = sc_llr_for_bit(path.llr, path.u_hat, phi, N)
                if self.frozen_bits[phi]:
                    path.pm += self._path_metric_penalty(llr_phi, 0)
                    path.u_hat[phi] = 0
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += self._path_metric_penalty(llr_phi, u)
                        child.u_hat[phi] = u
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths
                     if crc_check(p.u_hat[self.info_idx], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat, best.pm
