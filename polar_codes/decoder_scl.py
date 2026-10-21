"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from polar_ops import f_operation, g_operation

_FloatArray = np.ndarray
_ByteArray = np.ndarray


def _f(a, b):
    return f_operation(a, b)


def _g(a, b, u):
    return np.asarray(b + (1.0 - 2.0 * u.astype(np.float64)) * a, dtype=np.float64)


def _penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.uint8) % 2
    poly = 0x07 if crc_length == 8 else 0x8005
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) & 1) ^ int(bit)
        reg = (reg << 1) & mask
        if fb:
            reg ^= poly
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.uint8) % 2
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class _SCLDecoderCore:
    """极化树 SCL 译码核"""

    def __init__(self, frozen: np.ndarray, list_size: int) -> None:
        self.frozen = frozen
        self.list_size = list_size
        self.block_length = int(frozen.size)
        self.metrics: list[float] = [0.0]
        self.decisions: list[_ByteArray] = [np.zeros(self.block_length, dtype=np.uint8)]

    def decode(self, channel_llr) -> list[tuple[float, _ByteArray]]:
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        self._node([llr], 0, self.block_length)
        return sorted(zip(self.metrics, self.decisions), key=lambda path: path[0])

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
        new_metrics, new_decisions, betas, parent_map = [], [], [], []
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
        upper = [_f(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)
        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [_g(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
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
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        core = _SCLDecoderCore(self.frozen, self.list_size)
        paths = core.decode(llr_ch)
        if self.crc_length > 0:
            valid = [(m, d) for m, d in paths if crc_check(d, self.crc_length)]
            if valid:
                paths = valid
        pm, u_hat = paths[0]
        return u_hat.astype(int), pm
