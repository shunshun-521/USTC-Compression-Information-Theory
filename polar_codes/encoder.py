"""
极化码编码器
编码：x = G_N * u（Arikan 极化变换 F^{\\otimes n}）
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array(
        [int(format(i, f"0{n}b")[::-1], 2) for i in range(N)],
        dtype=np.int64,
    )


def _polar_transform_recursive(u):
    """递归极化变换（与 dit.coding._util.polar_transform 一致）"""
    n = len(u)
    if n == 1:
        return [int(u[0])]
    half = n // 2
    upper = [int(u[i]) ^ int(u[i + half]) for i in range(half)]
    return _polar_transform_recursive(upper) + _polar_transform_recursive(
        [int(x) for x in u[half:]]
    )


def polar_encode(u):
    """
    极化码编码：x = F^{\\otimes n} u。
    复杂度 O(N log N)。
    """
    u = np.asarray(u, dtype=np.int8)
    N = len(u)
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("N must be a power of 2")
    return np.array(_polar_transform_recursive(u), dtype=int)


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("encoder self-test OK:", x)
