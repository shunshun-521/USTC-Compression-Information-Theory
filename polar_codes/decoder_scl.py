"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _exact_f, g_operation, _frozen_mask, _penalty


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07 (x^8+x^2+x+1); CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 decisions 向量）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = _frozen_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        frozen = self.frozen
        L = self.list_size
        N = self.N

        metrics = [0.0]
        decisions = [np.zeros(N, dtype=np.uint8)]

        def leaf(llrs, index):
            if frozen[index]:
                for path, llr_val in enumerate(llrs):
                    metrics[path] += _penalty(float(llr_val[0]), 0)
                    decisions[path][index] = 0
                return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

            candidates = []
            for path, llr_val in enumerate(llrs):
                for bit in (0, 1):
                    candidates.append(
                        (metrics[path] + _penalty(float(llr_val[0]), bit), path, bit)
                    )
            candidates.sort(key=lambda x: x[0])
            kept = candidates[:L]

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

        def tree_node(llrs, base, length):
            if length == 1:
                return leaf(llrs, base)

            half = length // 2
            upper = [_exact_f(l[:half], l[half:]) for l in llrs]
            beta_up, map_up = tree_node(upper, base, half)

            lower = [
                g_operation(
                    llrs[map_up[p]][:half],
                    llrs[map_up[p]][half:],
                    beta_up[p],
                )
                for p in range(len(map_up))
            ]
            beta_lo, map_lo = tree_node(lower, base + half, half)

            beta_up = [beta_up[map_lo[p]] for p in range(len(map_lo))]
            betas = [
                np.concatenate([beta_up[p] ^ beta_lo[p], beta_lo[p]])
                for p in range(len(beta_lo))
            ]
            parent_map = [map_up[map_lo[p]] for p in range(len(map_lo))]
            return betas, parent_map

        tree_node([llr], 0, N)

        paths = sorted(zip(metrics, decisions), key=lambda x: x[0])
        if self.crc_length > 0:
            info_idx = np.where(~frozen)[0]
            for pm, u_hat in paths:
                payload = u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    return u_hat.astype(int), pm
        pm, u_hat = paths[0]
        return u_hat.astype(int), pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    ok = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_scl, u_sc)
        if np.array_equal(u_scl, u):
            ok += 1
    print(f"SCL L=1 matches SC: {ok}/50")
