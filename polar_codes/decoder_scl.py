"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from numpy.typing import NDArray


def f_boxplus(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """精确 log 域 f 运算（check node）"""
    return np.asarray(np.logaddexp(0.0, a + b) - np.logaddexp(a, b), dtype=np.float64)


def g_operation(
    a: NDArray[np.float64], b: NDArray[np.float64], u: NDArray
) -> NDArray[np.float64]:
    """g 运算：b + (1 - 2u) * a"""
    return np.asarray(b + (1.0 - 2.0 * np.asarray(u, dtype=np.float64)) * a, dtype=np.float64)


def _path_penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int64)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int64)
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    return reg == 0


class _SCLCore:
    """极化码 SCL 树形译码核心（Lazy Copy 通过 parent_map 实现）"""

    def __init__(self, frozen: NDArray[np.bool_], list_size: int):
        self.frozen = frozen
        self.list_size = list_size
        self.block_length = int(frozen.size)
        self.metrics: list[float] = [0.0]
        self.decisions: list[NDArray[np.uint8]] = [
            np.zeros(self.block_length, dtype=np.uint8)
        ]

    def decode(self, channel_llr) -> list[tuple[float, NDArray[np.uint8]]]:
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        self._node([llr], 0, self.block_length)
        return sorted(
            zip(self.metrics, self.decisions, strict=True), key=lambda x: x[0]
        )

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _path_penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _path_penalty(float(llr[0]), bit), path, bit)
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
        upper = [f_boxplus(llr[:half], llr[half:]) for llr in llrs]
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


def scl_decode_paths(channel_llr, frozen_bits, list_size):
    frozen = np.asarray(frozen_bits, dtype=bool)
    return _SCLCore(frozen, list_size).decode(channel_llr)


class SCLDecoder:
    """SCL / CA-SCL 译码器封装"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        paths = scl_decode_paths(llr_ch, self.frozen_bits, self.list_size)
        if self.crc_length > 0:
            valid = [
                (pm, u)
                for pm, u in paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                pm, u_hat = min(valid, key=lambda x: x[0])
                return u_hat.astype(int), pm
        pm, u_hat = paths[0]
        return u_hat.astype(int), pm
