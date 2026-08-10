"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _penalty


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    width = crc_length

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)

    crc_bits = np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    width = crc_length
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.metrics = [0.0]
        self.decisions = [np.zeros(N, dtype=int)]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=int)]
        self._node([llr_ch], 0, self.N)

        paths = list(zip(self.metrics, self.decisions))
        paths.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            valid = []
            for pm, u_hat in paths:
                info_bits = u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                best = min(valid, key=lambda x: x[0])
                return best[1].copy(), best[0]

        return paths[0][1].copy(), paths[0][0]

    def _leaf(self, llrs, index):
        if self.frozen_bits[index]:
            new_metrics = []
            new_decisions = []
            betas = []
            parent_map = []
            for path, llr in enumerate(llrs):
                new_metrics.append(self.metrics[path] + _penalty(float(llr[0]), 0))
                dec = self.decisions[path].copy()
                dec[index] = 0
                new_decisions.append(dec)
                betas.append(np.array([0], dtype=int))
                parent_map.append(path)
            self.metrics = new_metrics
            self.decisions = new_decisions
            return betas, parent_map

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((self.metrics[path] + _penalty(float(llr[0]), bit), path, bit))
        candidates.sort(key=lambda x: x[0])
        kept = candidates[:self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for pm, path, bit in kept:
            new_metrics.append(pm)
            dec = self.decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
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

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        return (
            [np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]]) for p in range(len(beta_lower))],
            [map_upper[map_lower[p]] for p in range(len(map_lower))],
        )


def verify_scl_equals_sc(N=64, K=32, num_frames=20, eb_n0_db=15.0):
    """验证 L=1 的 SCL 等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True


if __name__ == "__main__":
    print("SCL L=1 verification:", verify_scl_equals_sc())
