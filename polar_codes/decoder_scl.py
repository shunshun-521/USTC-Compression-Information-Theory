"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_GEN = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)   # x^8+x^2+x+1
_CRC16_GEN = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int
)  # CRC-16-IBM


def _crc_remainder_gf2(message, generator):
    """GF(2) 多项式长除求 CRC 余数。"""
    msg = np.asarray(message, dtype=int).copy()
    r = len(generator) - 1
    for i in range(len(msg) - r):
        if msg[i] == 1:
            msg[i : i + len(generator)] ^= generator
    return msg[-r:]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    generator = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    augmented = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder_gf2(augmented, generator)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    generator = _CRC8_GEN if crc_length == 8 else _CRC16_GEN
    remainder = _crc_remainder_gf2(bits, generator)
    return np.all(remainder == 0)


class _Path:
    """单条 SCL 路径，使用 lazy copy 共享数组。"""

    __slots__ = ("L", "B", "pm", "u_hat", "parent", "copied_L", "copied_B")

    def __init__(self, N, n, llr_ch, parent=None):
        self.parent = parent
        self.copied_L = False
        self.copied_B = False
        if parent is None:
            br = bit_reversal_permutation(N)
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=np.int8)
            self.L[:, 0] = llr_ch[br]
            self.pm = 0.0
            self.u_hat = np.zeros(N, dtype=np.int8)
        else:
            self.L = parent.L
            self.B = parent.B
            self.pm = parent.pm
            self.u_hat = parent.u_hat.copy()

    def ensure_copy(self):
        if self.parent is not None and not self.copied_L:
            self.L = self.L.copy()
            self.copied_L = True
        if self.parent is not None and not self.copied_B:
            self.B = self.B.copy()
            self.copied_B = True


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr_val, u_bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：
            u_hat: 长度 N 的估计源序列（最优路径）
            pm: 最优路径的度量值
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = self.br[i]
            new_paths = []

            for path in paths:
                path.ensure_copy()
                _update_llrs(path.L, path.B, l, self.n)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pen = self._pm_penalty(llr_val, 0)
                    path.pm += pen
                    path.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    _update_bits(path.B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _Path(self.N, self.n, llr_ch, parent=path)
                        child.ensure_copy()
                        child.pm = path.pm + self._pm_penalty(llr_val, u_bit)
                        child.B[l, self.n] = u_bit
                        child.u_hat[l] = u_bit
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_pos], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    errors_sc = errors_scl = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 0.01)

        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)

        if not np.array_equal(u, u_sc):
            errors_sc += 1
        if not np.array_equal(u, u_scl):
            errors_scl += 1

    print(f"L=1 SCL vs SC: sc_err={errors_sc}, scl_err={errors_scl}")
