"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_exact, g_operation
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（MSB-first，按比特输入）。"""
    info_bits = np.asarray(info_bits, dtype=np.uint8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.uint8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.uint8)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(expected, bits[-crc_length:])


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 decisions/metrics，LLR 按父路径索引）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        llr = llr[bit_reversal_permutation(self.N)]
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.uint8)]
        codewords, _ = self._node([llr], metrics, decisions, 0, self.N)
        paths = list(zip(metrics, decisions, codewords, strict=True))
        paths.sort(key=lambda p: p[0])

        if self.crc_length > 0:
            for metric, u_hat, _ in paths:
                if crc_check(u_hat, self.crc_length):
                    return u_hat.astype(np.int8), metric
        best = paths[0]
        return best[1].astype(np.int8), best[0]

    def _leaf(self, llrs, metrics, decisions, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                metrics[path] += _penalty(float(llr[0]), 0)
                decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((metrics[path] + _penalty(float(llr[0]), bit), path, bit))
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

    def _node(self, llrs, metrics, decisions, base, length):
        if length == 1:
            return self._leaf(llrs, metrics, decisions, base)

        half = length // 2
        upper = [f_exact(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, metrics, decisions, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, metrics, decisions, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
