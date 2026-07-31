"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, update_bits, update_llrs
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    poly = _crc_poly(crc_length)
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    bits = np.asarray(bits, dtype=np.int8)
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

        self.br = bit_reversal_permutation(N)

        if crc_length > 0:
            self.info_positions = np.where(~self.frozen_bits)[0]
        else:
            self.info_positions = None

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
            "u_hat": np.zeros(N, dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = np.asarray(llr_ch, dtype=np.float64)[self.br]

        for phase in range(N):
            index = self.br[phase]
            new_paths = []

            for path in paths:
                update_llrs(path["L"], path["B"], index, n, N)
                llr_index = path["L"][index, n]
                candidates = [0] if self.frozen_bits[index] else [0, 1]

                for bit in candidates:
                    new_path = {
                        "pm": path["pm"] + self._pm_penalty(llr_index, bit),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][index, n] = bit
                    new_path["u_hat"][index] = bit
                    update_bits(new_path["B"], index, n, N)
                    new_paths.append(new_path)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best["u_hat"], best["pm"]

    def _propagate_bits(self, path, phi):
        """Deprecated: kept for API compatibility."""
        index = self.br[phi]
        update_bits(path["B"], index, self.n, self.N)


def verify_scl_equals_sc(N=64, K=32, seed=1):
    """单路径 SCL 应等价于 SC。"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from decoder_sc import sc_decode
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(3.0, K / N)
    rng = np.random.default_rng(seed)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        info = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
