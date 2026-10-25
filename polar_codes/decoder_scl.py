"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation_exact, g_operation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_update_byte(reg, byte, poly, width):
    reg ^= byte << (width - 8)
    for _ in range(8):
        if reg & (1 << (width - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
        else:
            reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8) % 2
    if crc_length == 8:
        poly, width = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    nbytes = (len(info_bits) + 7) // 8
    padded = np.zeros(nbytes * 8, dtype=np.int8)
    padded[: len(info_bits)] = info_bits
    for i in range(nbytes):
        byte = 0
        for b in range(8):
            byte |= int(padded[i * 8 + b]) << (7 - b)
        reg = _crc_update_byte(reg, byte, poly, width)

    crc_bits = np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8) % 2
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _pm_penalty(llr, bit):
    """路径度量：与硬判决不一致时加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class _SCLCore:
    """SCL 树遍历（Lazy Copy）"""

    def __init__(self, frozen, list_size):
        self.frozen = np.asarray(frozen, dtype=bool)
        self.list_size = int(list_size)
        self.block_length = int(self.frozen.size)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        self._node([llr], 0, self.block_length)
        return sorted(zip(self.metrics, self.decisions, strict=True), key=lambda x: x[0])

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _pm_penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _pm_penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.uint8))
            parent_map.append(path)
        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation_exact(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map


class SCLDecoder:
    """SCL / CA-SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = int(N)
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.info_idx = np.flatnonzero(~self.frozen)

    def decode(self, llr_ch):
        paths = _SCLCore(self.frozen, self.list_size).decode(llr_ch)

        if self.crc_length > 0:
            for pm, u_hat in paths:
                payload = u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    return u_hat.astype(int), pm

        pm, u_hat = paths[0]
        return u_hat.astype(int), pm
