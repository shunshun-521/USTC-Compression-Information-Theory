"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np


def _f_exact(a, b):
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def _f_min_sum(a, b):
    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def _g(a, b, u):
    return b + (1.0 - 2.0 * u.astype(np.float64)) * a


def _penalty(llr, bit):
    """路径度量增量（越小越好）"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def _crc_remainder(data_bits, crc_length):
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for bit in data_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8) % 2
    reg = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8) % 2
    reg = _crc_remainder(bits, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制决策向量，LLR 按父路径索引映射）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, use_min_sum=True):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        if self.frozen.size != N:
            raise ValueError("frozen_bits length mismatch")
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.use_min_sum = use_min_sum
        self._f = _f_min_sum if use_min_sum else _f_exact

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._scl_decode(llr_ch)
        best_u, best_pm = self._select_path(paths)
        return best_u.astype(int), best_pm

    def _select_path(self, paths):
        """paths: list of (metric, u_hat)"""
        if self.crc_length <= 0:
            pm, u = paths[0]
            return u, pm

        passed = []
        for pm, u in paths:
            payload = u[~self.frozen]
            if crc_check(payload, self.crc_length):
                passed.append((pm, u))
        if passed:
            return passed[0][1], passed[0][0]
        return paths[0][1], paths[0][0]

    def _scl_decode(self, channel_llr):
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.int8)]
        _, _ = self._node([channel_llr], 0, self.N, metrics, decisions)
        paths = sorted(zip(metrics, decisions), key=lambda x: x[0])
        return paths

    def _leaf(self, llrs, index, metrics, decisions):
        if self.frozen[index]:
            for p, llr in enumerate(llrs):
                metrics[p] += _penalty(float(llr[0]), 0)
                decisions[p][index] = 0
            return [np.array([0], dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for p, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((metrics[p] + _penalty(float(llr[0]), bit), p, bit))
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, parent, bit in kept:
            new_metrics.append(metric)
            dec = decisions[parent].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.int8))
            parent_map.append(parent)

        metrics[:] = new_metrics
        decisions[:] = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length, metrics, decisions):
        if length == 1:
            return self._leaf(llrs, base, metrics, decisions)

        half = length // 2
        upper = [self._f(llr[:half], llr[half:]) for llr in llrs]
        beta_up, map_up = self._node(upper, base, half, metrics, decisions)

        a = [llrs[map_up[p]][:half] for p in range(len(map_up))]
        b = [llrs[map_up[p]][half:] for p in range(len(map_up))]
        lower = [_g(a[p], b[p], beta_up[p]) for p in range(len(beta_up))]
        beta_lo, map_lo = self._node(lower, base + half, half, metrics, decisions)

        beta_up = [beta_up[map_lo[p]] for p in range(len(map_lo))]
        betas = [
            np.concatenate([beta_up[p] ^ beta_lo[p], beta_lo[p]]) for p in range(len(beta_lo))
        ]
        parent_map = [map_up[map_lo[p]] for p in range(len(map_lo))]
        return betas, parent_map


def scl_decode_paths(llr_ch, frozen_bits, list_size=4, use_min_sum=True):
    """返回所有幸存路径 (metric, u_hat)"""
    dec = SCLDecoder(len(llr_ch), frozen_bits, list_size=list_size, use_min_sum=use_min_sum)
    return dec._scl_decode(np.asarray(llr_ch, dtype=np.float64))
