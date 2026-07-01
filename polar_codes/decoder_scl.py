"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _prepare_llr, _compute_llr, _b_check, _s_updater,
    f_operation, g_operation,
)
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数"""
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                break
        if crc_length == 16:
            if reg & (1 << 15):
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
        reg = 0
        for bit in bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        return reg == 0
    poly = CRC16_POLY
    reg = 0
    for bit in bits:
        reg ^= int(bit) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg == 0


class _PathState:
    """单条译码路径状态（Lazy Copy）"""

    __slots__ = ('llrs', 's', 'pm', 'u_hat', 'active')

    def __init__(self, n, N):
        self.llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
        self.s = np.full((n + 1, N), -1, dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _clone_path(self, path):
        new_path = _PathState(self.n, self.N)
        new_path.llrs = path.llrs.copy()
        new_path.s = path.s.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _path_metric_penalty(self, llr, bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        preferred = 0 if llr >= 0 else 1
        return 0.0 if bit == preferred else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = _prepare_llr(llr_ch)
        paths = [_PathState(self.n, self.N)]
        paths[0].llrs[self.n, :] = llr_ch

        for phi in range(self.N):
            new_paths = []

            for path in paths:
                if not path.active:
                    continue

                if self.frozen_bits[phi]:
                    llr_val = _compute_llr(0, phi, path.llrs, path.s)
                    path.pm += self._path_metric_penalty(llr_val, 0)
                    path.u_hat[phi] = 0
                    path.s[0, phi] = 0
                    path.llrs[0, phi] = np.inf
                    new_paths.append(path)
                else:
                    llr_val = _compute_llr(0, phi, path.llrs, path.s)

                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child.pm += self._path_metric_penalty(llr_val, bit)
                        child.u_hat[phi] = bit
                        child.s[0, phi] = bit
                        child.llrs[0, phi] = llr_val
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.u_hat.copy(), best.pm
