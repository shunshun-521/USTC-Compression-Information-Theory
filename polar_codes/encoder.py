"""
极化码编码器
编码：x = u * G_N，与因子图 SC/SCL 译码器一致的 stage-wise XOR 实现
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(format(i, f"0{n}b")[::-1], 2)
    return rev


def _gen_encode_indices(N: int) -> np.ndarray:
    """Sionna 风格的逐层 gather 索引（stage-wise XOR）。"""
    nb_stages = int(np.log2(N))
    ind_gather = np.ones((nb_stages, N + 1), dtype=np.int32) * N
    for s in range(nb_stages):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u, info_indices=None):
    """
    极化码编码。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位），或
        info_indices: 若提供，则 u 仅为 K 个信息比特并按 info_indices 映射。
    """
    if info_indices is not None:
        u_full = np.zeros(int(np.max(info_indices)) + 1 if len(info_indices) else 1, dtype=np.int8)
        N = len(u_full)
        # N must be inferred from max index + 1 — caller should pass full u instead
        raise ValueError("Pass full length-N vector u with frozen positions set to 0")

    u = np.asarray(u, dtype=np.uint8)
    N = len(u)
    n = int(np.log2(N))
    assert 2**n == N

    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_encode_indices(N)

    for s in range(n):
        ind_helper = ind_gather[s, :]
        x_add = x[ind_helper]
        x = np.bitwise_xor(x, x_add)

    return x[:N].astype(np.int8)


def polar_encode_info(info_bits: np.ndarray, info_indices: np.ndarray, N: int) -> np.ndarray:
    """将 K 个信息比特映射到 u 向量并编码。"""
    u = np.zeros(N, dtype=np.int8)
    u[info_indices] = info_bits
    return polar_encode(u)


if __name__ == "__main__":
    # 与 u @ G_N（含比特倒序）等价的经典测试向量
    u = np.array([1, 0, 1, 1])
    G = np.array([[1, 0, 0, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
    x_ref = (u @ G) % 2
    # stage-wise 编码与矩阵形式在倒序约定下一致；此处验证无噪链路
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    N = 4
    info = np.array([2, 3])
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    u4 = np.zeros(N, dtype=int)
    u4[info] = [1, 0]
    x = polar_encode(u4)
    llr = -compute_llr(bpsk_modulate(x), 1e-9)
    uh = sc_decode(llr, frozen)
    assert uh[info].tolist() == [1, 0], f"roundtrip failed: {uh[info]}"
    print("encoder/decoder roundtrip ok")
