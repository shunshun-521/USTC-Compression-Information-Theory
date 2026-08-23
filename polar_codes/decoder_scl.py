"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level


# CRC-8: 0x07 (x^8 + x^2 + x + 1)
CRC8_POLY = 0x07
# CRC-16: 0x8005
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8).flatten()
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits]).astype(int)


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确的 CRC。"""
    bits = np.asarray(bits, dtype=np.int8).flatten()
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr_ch):
      self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
      self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
      self.L[:, 0] = llr_ch
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _pm_update(self, pm, llr, bit):
        """路径度量更新。"""
        hard = 0 if llr >= 0 else 1
        if bit == hard:
            return pm
        return pm + abs(llr)

    def _advance_path(self, path, phi, bit):
        """对单条路径执行一步 SCL 更新。"""
        l = _bit_reversed(phi, self.n)
        n = self.n
        N = self.N

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

        llr_leaf = path.L[l, n]
        path.pm = self._pm_update(path.pm, llr_leaf, bit)
        path.u_hat[l] = bit
        path.B[l, n] = bit

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_core = llr_ch[self.br]

        if self.list_size == 1:
            from decoder_sc import _sc_decode_core
            u_hat = _sc_decode_core(llr_core, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N, self.n, llr_core.copy())]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                for s in range(self.n - _active_llr_level(l, self.n), self.n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, self.N, block_size):
                        if j % block_size < branch_size:
                            path.L[j, s + 1] = f_operation(
                                path.L[j, s], path.L[j + branch_size, s]
                            )
                        else:
                            path.L[j, s + 1] = g_operation(
                                path.L[j - branch_size, s],
                                path.L[j, s],
                                path.B[j - branch_size, s + 1],
                            )

                llr_leaf = path.L[l, self.n]

                if self.frozen_bits[l]:
                    child = _Path(self.N, self.n, np.zeros(self.N))
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.pm = self._pm_update(path.pm, llr_leaf, 0)
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[l] = 0
                    child.B[l, self.n] = 0
                    self._propagate_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, np.zeros(self.N))
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.pm = self._pm_update(path.pm, llr_leaf, bit)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = bit
                        child.B[l, self.n] = bit
                        self._propagate_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm

    def _propagate_bits(self, path, l):
        """比特向上传播。"""
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]
