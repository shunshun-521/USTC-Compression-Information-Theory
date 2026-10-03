"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np


def _f_kernel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def _g_kernel(a, b, u):
    u = np.asarray(u, dtype=np.float64)
    return b + (1.0 - 2.0 * u) * a


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（MSB 先行）"""
    info_bits = np.asarray(info_bits, dtype=np.uint8) % 2
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.uint8) % 2
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


class _SCLDecoderCore:
    """SCL 译码内核（列表路径 + 父路径映射）"""

    def __init__(self, frozen, list_size, crc_length=0):
        self.frozen = np.asarray(frozen, dtype=bool)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.block_length = self.frozen.size
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        _, _ = self._node([llr], 0, self.block_length)
        paths = list(zip(self.metrics, self.decisions, strict=True))
        paths.sort(key=lambda x: x[0])
        return paths

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
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
        upper = [_f_kernel(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [_g_kernel(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map


class SCLDecoder:
    """SCL / CA-SCL 译码器封装"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        fb = np.asarray(frozen_bits)
        if np.issubdtype(fb.dtype, np.integer):
            self.frozen = fb.astype(bool)
        else:
            self.frozen = fb.astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        core = _SCLDecoderCore(self.frozen, self.list_size, self.crc_length)
        paths = core.decode(llr_ch)
        if self.crc_length > 0:
            info_pos = np.flatnonzero(~self.frozen)
            for pm, u_hat in paths:
                payload = u_hat[info_pos]
                if crc_check(payload, self.crc_length):
                    return u_hat.astype(int), pm
        return paths[0][1].astype(int), paths[0][0]


def sc_decode_via_scl(llr_ch, frozen_bits):
    """L=1 时等价于 SC 译码"""
    dec = SCLDecoder(len(llr_ch), frozen_bits, list_size=1, crc_length=0)
    u_hat, pm = dec.decode(llr_ch)
    return u_hat
