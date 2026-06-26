"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _upper_llr_min_sum,
    _lower_llr_min_sum,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    """对比特序列做 CRC 除法，返回余数"""
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_process(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("L", "C", "pm", "u_hat", "active")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.C = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _path_metric_penalty(self, llr, u):
        """路径度量惩罚：与 LLR 硬判决不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        N, n = self.N, self.n

        paths = [_Path(N, n, llr_ch.copy())]

        for i in range(N):
            l = _bit_reversed(i, n)
            is_frozen = l in self.frozen_set
            candidates = []

            for pidx, path in enumerate(paths):
                if not path.active:
                    continue

                start_s = n - _active_llr_level(l, n)
                for s in range(start_s, n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            path.L[j, s + 1] = _upper_llr_min_sum(
                                path.L[j, s], path.L[j + branch_size, s]
                            )
                        else:
                            path.L[j, s + 1] = _lower_llr_min_sum(
                                path.L[j, s],
                                path.L[j - branch_size, s],
                                path.C[j - branch_size, s + 1],
                            )

                llr_bit = path.L[l, n]

                if is_frozen:
                    pen = self._path_metric_penalty(llr_bit, 0)
                    path.pm += pen
                    path.C[l, n] = 0
                    path.u_hat[l] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        new_path = self._fork_path(path)
                        pen = self._path_metric_penalty(llr_bit, u)
                        new_path.pm += pen
                        new_path.C[l, n] = u
                        new_path.u_hat[l] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            for p in paths:
                p.active = True

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_positions = np.where(self.frozen_bits == 0)[0]
                payload = path.u_hat[info_positions]
                if crc_check(payload, self.crc_length):
                    return path.u_hat.copy(), path.pm

        return paths[0].u_hat.copy(), paths[0].pm

    def _fork_path(self, path):
        """Lazy copy：仅复制必要数组"""
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.L = path.L.copy()
        new_path.C = path.C.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        end_s = self.n - _active_bit_level(l, self.n) + 1
        for s in range(self.n, end_s - 1, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.C[j - branch_size, s - 1] = (
                        path.C[j, s] ^ path.C[j - branch_size, s]
                    )
                    path.C[j, s - 1] = path.C[j, s]
