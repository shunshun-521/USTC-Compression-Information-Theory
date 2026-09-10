"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation


CRC_CONFIG = {
    8: (0x07, 0x00),
    16: (0x1021, 0x0000),
}


def _crc_compute(info_bits, crc_length):
    """CRC 按字节计算（CRC-8: poly=0x07; CRC-16: poly=0x1021）。"""
    poly, init = CRC_CONFIG[crc_length]
    mask = (1 << crc_length) - 1
    top_bit = 1 << (crc_length - 1)
    crc = init & mask

    data = np.packbits(np.asarray(info_bits, dtype=np.uint8))
    for raw_byte in data.tobytes():
        crc ^= raw_byte << (crc_length - 8)
        for _ in range(8):
            if crc & top_bit:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    value = _crc_compute(info_bits, crc_length)
    crc_bits = np.array(
        [(value >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=int)
    msg = bits[:-crc_length]
    expected = crc_encode(msg, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr, bit):
    """路径度量惩罚（log-domain）"""
    return float(np.logaddexp(0.0, -(1 - 2 * bit) * llr))


class SCLDecoder:
    """
    SCL 译码器（递归树 + Lazy Copy 路径映射）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.flatnonzero(~self.frozen_bits)
        )
        self.metrics = [0.0]
        self.decisions = [np.zeros(N, dtype=int)]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=int)]
        _, _ = self._node([llr_ch], 0, self.N)

        if self.crc_length > 0:
            best_crc = None
            best_pm = float("inf")
            for pm, u_hat in zip(self.metrics, self.decisions, strict=True):
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length) and pm < best_pm:
                    best_crc = u_hat.copy()
                    best_pm = pm
            if best_crc is not None:
                return best_crc, best_pm

        best_idx = int(np.argmin(self.metrics))
        return self.decisions[best_idx].copy(), self.metrics[best_idx]

    def _leaf(self, llrs, index):
        if self.frozen_bits[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _pm_penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=int) for _ in llrs], list(range(len(llrs)))

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
            betas.append(np.array([bit], dtype=int))
            parent_map.append(path)

        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        lower = [
            g_operation(
                llrs[map_upper[p]][:half],
                llrs[map_upper[p]][half:],
                beta_upper[p],
            )
            for p in range(len(map_upper))
        ]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(map_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
