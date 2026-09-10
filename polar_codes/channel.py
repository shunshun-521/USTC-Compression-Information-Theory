"""
AWGN 信道 + BPSK 调制/解调
与 polarcodes Permuted SCD 配对：0 -> -sqrt(Es), 1 -> +sqrt(Es)
"""
import numpy as np


def _normalized_es(eb_n0_db, rate):
    return rate * (10.0 ** (eb_n0_db / 10.0))


def bpsk_modulate(x, eb_n0_db=None, rate=0.5):
    """将二进制码字 x (0/1) 映射为 BPSK 符号"""
    x = np.asarray(x, dtype=np.float64)
    if eb_n0_db is None:
        return 1.0 - 2.0 * x
    es = _normalized_es(eb_n0_db, rate)
    return 2.0 * (x - 0.5) * np.sqrt(es)


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return s + noise


def compute_llr(y, sigma=None, eb_n0_db=None, rate=0.5):
    """
    计算 BPSK-AWGN 信道 LLR（与 polarcodes 一致）。
    LLR = -2 * y * Es / No，No = sigma^2，Es = rate * 10^(Eb/N0/10)
    """
    y = np.asarray(y, dtype=np.float64)
    if eb_n0_db is not None:
        es = _normalized_es(eb_n0_db, rate)
        return -2.0 * y * es
    if sigma is None:
        raise ValueError('compute_llr 需要 sigma 或 eb_n0_db')
    es = 1.0 / (sigma ** 2)
    return -2.0 * y * es


def eb_n0_to_sigma(eb_n0_db, rate):
    """Eb/N0 (dB) -> AWGN 标准差（单位噪声功率 No=1）"""
    es = _normalized_es(eb_n0_db, rate)
    return 1.0 / np.sqrt(es)


def hard_decision_llr(llr):
    return (llr < 0).astype(int)
