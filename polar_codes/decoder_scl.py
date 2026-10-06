"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive, _frozen_to_bool


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位（MSB-first，标准 LFSR）。"""
    info_bits = np.asarray(info_bits, dtype=np.uint8) % 2
    poly = _crc_poly(crc_length)
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
    bits = np.asarray(bits, dtype=np.uint8) % 2
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _pm_update(llr_val, bit):
    preferred = 0 if llr_val >= 0 else 1
    return 0.0 if int(bit) == preferred else float(abs(llr_val))


class SCLDecoder:
    """SCL 译码器（路径复制 + 列表裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = int(N)
        self.frozen = _frozen_to_bool(frozen_bits)
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_positions = np.flatnonzero(~self.frozen)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        dec = _SCLCore(self.frozen, self.list_size)
        paths = dec.run(llr_ch)
        paths.sort(key=lambda t: t[0])

        if self.crc_length > 0:
            for pm, u_hat in paths:
                payload = u_hat[self.info_positions]
                if crc_check(payload, self.crc_length):
                    return u_hat.astype(int), pm

        pm, u_hat = paths[0]
        return u_hat.astype(int), pm


class _SCLCore:
    def __init__(self, frozen, list_size):
        self.frozen = frozen
        self.list_size = list_size
        self.N = frozen.size
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int8)]

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _pm_update(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append(
                    (self.metrics[path] + _pm_update(float(llr[0]), bit), path, bit)
                )
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = self.decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.int8))
            parent_map.append(path)
        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_u, map_u = self._node(upper, base, half)

        lowers = []
        for p, mp in enumerate(map_u):
            a = llrs[mp][:half]
            b = llrs[mp][half:]
            lowers.append(g_operation(a, b, beta_u[p]))
        beta_l, map_l = self._node(lowers, base + half, half)

        beta_u = [beta_u[map_l[p]] for p in range(len(map_l))]
        betas = [
            np.concatenate([beta_u[p] ^ beta_l[p], beta_l[p]]) for p in range(len(beta_l))
        ]
        parent_map = [map_u[map_l[p]] for p in range(len(map_l))]
        return betas, parent_map

    def run(self, llr_ch):
        betas, _ = self._node([llr_ch], 0, self.N)
        return list(zip(self.metrics, self.decisions, strict=True))


def scl_equals_sc(N, frozen_bits, trials=20, seed=0):
    """校验 L=1 时 SCL 与 SC 等价。"""
    rng = np.random.default_rng(seed)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(trials):
        llr = rng.normal(0, 2.0, N)
        u_sc = sc_decode_recursive(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
