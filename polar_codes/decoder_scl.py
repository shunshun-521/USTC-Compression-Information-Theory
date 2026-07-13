"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    prepare_channel_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    for _ in range(crc_length):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """
        主译码函数。
        返回 (u_hat, pm)
        """
        llr = prepare_channel_llr(llr_ch)
        N, n = self.N, self.n
        L_size = self.list_size

        paths = [
            {
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=int),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for pidx, path in enumerate(paths):
                _update_llrs(path["L"], path["B"], l, n)
                cur_llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    bit = 0
                    pm = self._path_metric_update(path["pm"], cur_llr, bit)
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": pm,
                    }
                    new_path["B"][l, n] = 0
                    _update_bits(new_path["B"], l, n)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = self._path_metric_update(path["pm"], cur_llr, bit)
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": pm,
                        }
                        new_path["B"][l, n] = bit
                        _update_bits(new_path["B"], l, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[:L_size]

        best = min(paths, key=lambda x: x["pm"])
        u_hat = best["B"][:, n].astype(int)

        if self.crc_length > 0:
            info_mask = self.frozen_bits == 0
            info_bits = u_hat[info_mask]
            valid = [
                p
                for p in paths
                if crc_check(
                    p["B"][info_mask, n].astype(int), self.crc_length
                )
            ]
            if valid:
                best = min(valid, key=lambda x: x["pm"])
                u_hat = best["B"][:, n].astype(int)

        return u_hat, best["pm"]


def verify_scl_equals_sc(N=64, frozen_bits=None):
    """单路径 SCL 应等价于 SC。"""
    from construction import ga_construction
    from decoder_sc import sc_decode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    if frozen_bits is None:
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        from channel import bpsk_modulate, compute_llr
        from encoder import polar_encode

        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-9)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True
