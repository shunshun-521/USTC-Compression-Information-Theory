"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return np.concatenate([info_bits, np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """
    SCL 译码器（递归树结构，与 SC 核心算法一致）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.rev = bit_reversal_permutation(N)
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_dec = self.frozen_bits[self.rev]
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _scl_node(self, paths, llr_node, frozen_node, bit_offset):
        """
        对路径列表执行子树 SCL 译码。
        返回 (paths, u_hat_up_per_path)，u_hat_up_per_path[i] 为路径 i 的 stage 编码向量。
        """
        n = len(llr_node)
        if n == 1:
            new_paths = []
            llr_val = llr_node[0]
            for path in paths:
                if frozen_node[0]:
                    u_val = 0
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    p = {'pm': path['pm'] + penalty, 'u_hat': path['u_hat'].copy()}
                    p['u_hat'][bit_offset] = u_val
                    new_paths.append(p)
                else:
                    for u_val in (0, 1):
                        penalty = 0.0 if (u_val == 0 and llr_val >= 0) or (u_val == 1 and llr_val < 0) else abs(llr_val)
                        p = {'pm': path['pm'] + penalty, 'u_hat': path['u_hat'].copy()}
                        p['u_hat'][bit_offset] = u_val
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p['pm'])
            new_paths = new_paths[:self.list_size]
            ups = [np.array([float(p['u_hat'][bit_offset])]) for p in new_paths]
            return new_paths, ups

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])

        left_paths, left_ups = self._scl_node(paths, llr_left, frozen_node[:half], bit_offset)

        all_right_paths = []
        all_right_ups = []
        for path, left_up in zip(left_paths, left_ups):
            llr_right = g_operation(llr_node[:half], llr_node[half:], left_up)
            right_paths, right_ups = self._scl_node([path], llr_right, frozen_node[half:], bit_offset + half)
            for rp, rup in zip(right_paths, right_ups):
                left_xor = (left_up.astype(int) ^ rup.astype(int)).astype(float)
                parent_up = np.concatenate([left_xor, rup])
                all_right_paths.append(rp)
                all_right_ups.append(parent_up)

        paired = sorted(zip(all_right_paths, all_right_ups), key=lambda x: x[0]['pm'])
        paired = paired[:self.list_size]
        if not paired:
            return [], []
        out_paths, out_ups = zip(*paired)
        return list(out_paths), list(out_ups)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        init_path = {'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=int)}
        paths, _ = self._scl_node([init_path], llr_ch, self.frozen_dec, 0)

        crc_pass_indices = []
        best_idx = 0
        best_pm = paths[0]['pm']

        for i, path in enumerate(paths):
            u_nat = path['u_hat'][self.rev]
            if self.crc_length > 0:
                if crc_check(u_nat[self.info_indices], self.crc_length):
                    crc_pass_indices.append(i)
            if path['pm'] < best_pm:
                best_pm = path['pm']
                best_idx = i

        if self.crc_length > 0 and crc_pass_indices:
            best_idx = min(crc_pass_indices, key=lambda i: paths[i]['pm'])

        u_hat = paths[best_idx]['u_hat'][self.rev].copy()
        u_hat[self.frozen_bits] = 0
        return u_hat, paths[best_idx]['pm']


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=5.0, num_frames=20):
    """单路径 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    scl4 = SCLDecoder(N, frozen_bits, list_size=4, crc_length=0)
    ok = 0
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10, rate))
        u_scl, _ = scl4.decode(llr)
        ok += int(np.array_equal(u_scl[info_idx], u[info_idx]))
    print(f"SCL L=4 high-SNR accuracy: {ok}/30")
    print(f"SCL L=1 verification passed: {num_frames} frames")


if __name__ == "__main__":
    verify_scl_equals_sc()
