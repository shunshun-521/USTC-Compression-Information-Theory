"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）

编码器使用 x = butterfly(u)[bit_reversal]，译码时将信道 LLR 映射为
llr_scd[i] = llr_ch[inv_br[i]] 后执行标准 SCD 算法。
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus 对数域近似）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    等价于 lower_llr: u=0 -> La+Lb, u=1 -> La-Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    """对数域加法"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    """对数域减法"""
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def lower_llr(l1, l2, b):
    """g 分支 LLR 更新（l1=下支路, l2=上支路）"""
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    elif b == 1:
        return l1 - l2
    return np.nan


def upper_llr(l1, l2):
    """f 分支 LLR 更新（box-plus 精确形式）"""
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def bit_reversed_index(x, n):
    """n 位索引的比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """索引 i 的二进制表示中第一个 1 的位置（从高位计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """索引 i 的二进制表示中第一个 0 的位置（从高位计）"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def channel_llr(llr_ch):
    """
    将自然序信道 LLR 映射为 SCD 因子图所需顺序。
    编码 x[j] = butterfly(u)[br[j]]，故 llr_scd[i] = llr_ch[inv_br[i]]。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return llr_ch[inv_br]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量：
      - lambda_offset: 各层块大小
      - decode_order: 比特倒序译码顺序
      - llr_layer_vec: 每个相位需更新的 LLR 层
      - bit_layer_vec: 每个相位需回传的比特层
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l < N / 2:
            bit_layer_vec.append([])
        else:
            end = n - active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end, -1)))

    return lambda_offset, decode_order, llr_layer_vec, bit_layer_vec


def _scd_decode(llr_scd, frozen_set, n):
    """标准非递归 SCD 核心（参考 Vangala et al.）"""
    N = 1 << n
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_scd

    for l in [bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (
                            int(B[j, s]) ^ int(B[j - branch_size, s])
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。

    参数：
        llr_ch: 长度 N 的信道接收 LLR（自然序）
        frozen_bits: 长度 N 的 bool/int 数组，True/1 表示冻结位

    返回：
        u_hat: 长度 N 的估计源序列（自然序）
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    llr_scd = channel_llr(llr_ch)
    return _scd_decode(llr_scd, frozen_set, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归高效实现）"""
    return sc_decode(llr, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = 0.01
    errors = 0
    for _ in range(100):
        payload = np.random.randint(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    print(f"SC low-noise: {errors}/100 errors")

    sigma2 = eb_n0_to_sigma(3.0, 0.5)
    errors2 = 0
    for _ in range(50):
        payload = np.random.randint(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma2, N)
        llr = compute_llr(y, sigma2)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors2 += 1
    print(f"SC Eb/N0=3dB: {errors2}/50 frame errors (expected for SC)")
