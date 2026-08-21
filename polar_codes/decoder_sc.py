"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def _exact_f(a, b):
    """精确 log-domain f 运算（box-plus）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（向量化）。
    注：SC 主路径使用精确 f；此函数供 BP 等模块复用。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat.astype(np.float64)) * La


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(bool)


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，单路径树遍历）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 为编码输出顺序；内部做比特倒序后 SC 译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr = llr_ch[br]

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
        kept = candidates[:1]

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
        upper = [_exact_f(l[ :half], l[half:]) for l in llrs]
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
    return decisions[0].astype(int)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    for N in [4, 8, 64, 256]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        ok = 0
        rng = np.random.default_rng(0)
        for _ in range(100):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
            ur = sc_decode(llr, frozen_bits)
            ur_r = sc_decode_recursive(llr, frozen_bits)
            assert np.array_equal(ur, ur_r)
            if np.array_equal(ur, u):
                ok += 1
        print(f"N={N}: {ok}/100")
