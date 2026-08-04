"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation
from encoder import bit_reversal_permutation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length, reg=0):
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
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    reg = _crc_process(padded, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


def _pm_penalty(llr, bit):
    """路径度量惩罚（log-domain）"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """SCL 译码器（递归实现 + 路径裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr = llr_ch[rev]

        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.int32)]
        _, _ = self._node([llr], 0, self.N, metrics, decisions)

        paths = list(zip(metrics, decisions, strict=True))
        paths.sort(key=lambda p: p[0])

        if self.crc_length > 0:
            crc_ok = [
                p for p in paths
                if crc_check(p[1][self.info_indices], self.crc_length)
            ]
            if crc_ok:
                paths = crc_ok

        best_pm, best_u = paths[0]
        return best_u.copy(), best_pm

    def _leaf(self, llrs, index, metrics, decisions):
        if self.frozen_bits[index]:
            for path, llr in enumerate(llrs):
                metrics[path] += _pm_penalty(float(llr[0]), 0)
                decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (metrics[path] + _pm_penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.uint8))
            parent_map.append(path)

        metrics[:] = new_metrics
        decisions[:] = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length, metrics, decisions):
        if length == 1:
            return self._leaf(llrs, base, metrics, decisions)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half, metrics, decisions)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half, metrics, decisions)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
