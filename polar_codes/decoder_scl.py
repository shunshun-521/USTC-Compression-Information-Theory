"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），PSC 风格非递归实现
"""
import math
import numpy as np
from decoder_sc import (
    f_operation, g_operation, precompute_sc_indices,
    _frozen_mask, bit_reversed,
)

# ==================== CRC 工具 ====================

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_serial(bits, poly, crc_length):
    """串行 CRC 计算"""
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_serial(msg, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_serial(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================


class Path:
    """单条译码路径"""

    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch.copy()

    def copy(self):
        new_path = Path.__new__(Path)
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（PSC 风格）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = _frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices
        self.decode_order, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _update_llr(self, path, step, l):
        for s in self.llr_layer_vec[step]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, step, l, u_val):
        path.u_hat[l] = u_val
        path.B[l, self.n] = u_val
        if l < self.N // 2:
            return
        for s in self.bit_layer_vec[step]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = np.array([bit_reversed(i, self.n) for i in range(self.N)])
        llr_aligned = llr_ch[rev]
        paths = [Path(self.N, self.n, llr_aligned)]

        for step, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                self._update_llr(path, step, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    child = path.copy()
                    child.pm += self._path_metric_penalty(llr, 0)
                    self._update_bits(child, step, l, 0)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = path.copy()
                        child.pm += self._path_metric_penalty(llr, u)
                        self._update_bits(child, step, l, u)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0 and self.info_indices is not None:
            valid = []
            for p in paths:
                info_bits = p.u_hat[np.sort(self.info_indices)]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p.pm) if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
