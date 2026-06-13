"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import sc_decode, _active_bit_level, _active_llr_level, _update_llrs
from encoder import bit_reversal_permutation


CRC_GENERATOR = {
    8: [1, 0, 0, 0, 0, 1, 1, 1, 1],
    16: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
}


def _poly_remainder(bits, generator):
    msg = [int(b) for b in bits]
    while len(msg) >= len(generator):
        if msg[0] == 1:
            for i in range(len(generator)):
                msg[i] ^= generator[i]
        msg.pop(0)
    return msg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    generator = CRC_GENERATOR[crc_length]
    remainder = _poly_remainder(np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]), generator)
    return np.concatenate([info_bits, np.array(remainder, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    generator = CRC_GENERATOR[crc_length]
    remainder = _poly_remainder(bits, generator)
    return all(x == 0 for x in remainder)


def _update_bits_path(B, l, n, N):
    """单路径比特回传"""
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch
        paths.append({"pm": 0.0, "L": L0, "B": B0, "u": np.zeros(N, dtype=np.int8)})

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n)
                llr0 = L[l, n]

                if self.frozen_bits[l]:
                    candidates = [(0, path["pm"] + (0.0 if llr0 >= 0 else abs(llr0)))]
                else:
                    candidates = [
                        (0, path["pm"] + (0.0 if llr0 >= 0 else abs(llr0))),
                        (1, path["pm"] + (0.0 if llr0 < 0 else abs(llr0))),
                    ]

                for bit, pm in candidates:
                    L_copy = path["L"].copy()
                    B_copy = path["B"].copy()
                    u_copy = path["u"].copy()
                    u_copy[l] = 0 if self.frozen_bits[l] else bit
                    B_copy[l, n] = u_copy[l]
                    _update_bits_path(B_copy, l, n, N)
                    new_paths.append({"pm": pm, "L": L_copy, "B": B_copy, "u": u_copy})

            order = np.argsort([p["pm"] for p in new_paths])
            keep = order[: self.list_size]
            paths = [new_paths[i] for i in keep]

        if self.crc_length > 0:
            crc_pass = [
                i
                for i, p in enumerate(paths)
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            best = min(crc_pass, key=lambda i: paths[i]["pm"]) if crc_pass else int(
                np.argmin([p["pm"] for p in paths])
            )
        else:
            best = int(np.argmin([p["pm"] for p in paths]))

        return paths[best]["u"], paths[best]["pm"]


def run_scl_self_test():
    """L=1 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(6.0, K / N)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
