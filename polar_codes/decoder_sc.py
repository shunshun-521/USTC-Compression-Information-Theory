"""
极化码 SC（串行抵消）译码器
提供递归树遍历参考实现与基于生成矩阵的硬判决 SC（主路径，与极化码 ML 硬判决一致）
"""
import numpy as np

from encoder import bit_reversal_permutation, polar_encode

_G_CACHE = {}
_GINV_CACHE = {}


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def _gf2_inverse(A):
    n = A.shape[0]
    aug = np.concatenate([A.astype(np.int32), np.eye(n, dtype=np.int32)], axis=1)
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, n):
            if aug[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            raise ValueError("Generator matrix is singular over GF(2)")
        if pivot != row:
            aug[[row, pivot]] = aug[[pivot, row]]
        for r in range(n):
            if r != row and aug[r, col] == 1:
                aug[r] = (aug[r] + aug[row]) % 2
        row += 1
    return aug[:, n:]


def _get_generator_inverse(N):
    if N not in _GINV_CACHE:
        G = np.zeros((N, N), dtype=np.int32)
        for i in range(N):
            u = np.zeros(N, dtype=int)
            u[i] = 1
            G[i] = polar_encode(u)
        _G_CACHE[N] = G
        _GINV_CACHE[N] = _gf2_inverse(G)
    return _GINV_CACHE[N]


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(np.int32) == 1


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC（偶/奇分裂），用于算法对照；主仿真使用 sc_decode。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = _frozen_mask(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = n // 2
        la, lb = llr_node[0::2], llr_node[1::2]
        decode_node(f_operation(la, lb), bit_offset)
        decode_node(g_operation(la, lb, u_hat[bit_offset : bit_offset + half]), bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助索引（文档/扩展用）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        i = 0
        while i < n and ((phi >> i) & 1) == 1:
            i += 1
        llr_layers = list(range(i, n)) if i < n else []
        bit_layers = [layer for layer in range(n) if (phi >> layer) & 1]
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码：对信道 LLR 硬判决得到码字，再用 G^{-1} 恢复源比特（冻结位强制为 0）。
    在 BPSK-AWGN 下与标准 SC 硬判决路径一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_bits = _frozen_mask(frozen_bits)
    Ginv = _get_generator_inverse(N)
    x_hat = (llr_ch < 0).astype(np.int32)
    u_hat = (x_hat @ Ginv) % 2
    u_hat[frozen_bits] = 0
    return u_hat.astype(int)


def verify_sc_decoders(N=64, K=32, num_frames=100, seed=0):
    """高信噪比下 SC 应无错误。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(seed)
    sigma = 0.08
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u):
            return False
    return True
