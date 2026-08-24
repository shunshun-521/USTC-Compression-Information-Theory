"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _SCDState,
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    """CRC LFSR 处理，返回最终寄存器值"""
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


def _path_metric_penalty(llr, u):
    """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
    preferred = 0 if llr >= 0 else 1
    return 0.0 if u == preferred else abs(llr)


class _Path:
    """SCL 单条路径"""

    __slots__ = ("state", "pm", "u_hat")

    def __init__(self, N, n, frozen_bits, llr_ch):
        self.state = _SCDState(N, frozen_bits, llr_ch.copy())
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, self.frozen_bits, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                path.state.update_llrs(l)
                llr = path.state.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if self.frozen_bits[l]:
                    u = 0
                    new_pm = path.pm + _path_metric_penalty(llr, u)
                    candidates.append((new_pm, path, u))
                else:
                    for u in (0, 1):
                        new_pm = path.pm + _path_metric_penalty(llr, u)
                        candidates.append((new_pm, path, u))

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u in survivors:
                child = _Path(self.N, self.n, self.frozen_bits, llr_ch)
                child.state.L = parent.state.L.copy()
                child.state.B = parent.state.B.copy()
                child.pm = pm
                child.u_hat = parent.u_hat.copy()

                child.state.B[l, self.n] = u
                child.u_hat[l] = u
                child.state.update_bits(l)
                new_paths.append(child)

            paths = new_paths

        # 选择最优路径
        crc_pass = []
        for p in paths:
            info_bits = p.u_hat[self.info_indices]
            if self.crc_length > 0:
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        pool = crc_pass if crc_pass else paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
