"""
AWGN 信道 + BPSK 调制/解调

用户接口（规范约定）：
  调制 0 -> +1, 1 -> -1
  LLR(y) = 2y/sigma^2, sigma = eb_n0_to_sigma(Eb/N0, R)

仿真内部使用 Es 缩放 + 固定 N0=1（与 SCD 译码器匹配）：
  0 -> -sqrt(Es), 1 -> +sqrt(Es), LLR = -2y*sqrt(Es)
"""
import numpy as np

_N0 = 1.0


def eb_n0_to_sigma(eb_n0_db, rate):
    """Eb/N0 (dB) -> AWGN 标准差 sigma = 1/sqrt(2R*10^{Eb/N0/10})"""
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def eb_n0_to_es(eb_n0_db, rate):
    """归一化符号能量 Es = R * Eb/N0（线性）"""
    return rate * (10.0 ** (eb_n0_db / 10.0))


def bpsk_modulate(x, es=None):
    """BPSK 调制；传入 es 时使用 Es 缩放（仿真默认）"""
    x = np.asarray(x, dtype=np.float64)
    if es is None:
        return 1.0 - 2.0 * x
    return 2.0 * (x - 0.5) * np.sqrt(es)


def awgn_channel(s, sigma=None, rng=None, es=None):
    """AWGN 信道；仿真路径使用固定 N0=1"""
    if rng is None:
        rng = np.random.default_rng()
    s = np.asarray(s, dtype=np.float64)
    noise_std = np.sqrt(_N0 / 2.0) if es is not None else sigma
    noise = rng.normal(0.0, noise_std, size=s.shape)
    return s + noise


def compute_llr(y, sigma=None, es=None):
    """信道 LLR"""
    y = np.asarray(y, dtype=np.float64)
    if es is not None:
        return -2.0 * y * np.sqrt(es) / _N0
    return 2.0 * y / (sigma ** 2)


def prepare_decoder_llr(llr_ch, N):
    return np.asarray(llr_ch, dtype=np.float64)


def map_decoder_bits_to_natural(u_hat_dec, N):
    return np.asarray(u_hat_dec, dtype=int)


def prepare_frozen_bits_decoder(frozen_nat, N):
    return np.asarray(frozen_nat, dtype=bool)
