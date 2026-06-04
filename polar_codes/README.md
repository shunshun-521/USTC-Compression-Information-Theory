# 极化码编译码仿真

纯 Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/BP 译码与蒙特卡洛仿真。

## 目录

- `construction.py` — 高斯近似（GA）构造
- `encoder.py` — O(N log N) XOR 蝶形编码
- `decoder_sc.py` — 递归与非递归 SC 译码
- `decoder_scl.py` — SCL / CA-SCL（CRC-8/16）
- `decoder_bp.py` — BP（min-sum，早停）
- `channel.py` — BPSK-AWGN
- `simulation.py` — 蒙特卡洛主循环
- `utils.py` — CSV、绘图、容量限
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py          # 校验 GA 构造
python run_exp1.py              # SC
python run_exp2.py              # SCL / CA-SCL
python run_exp3.py              # BP 对比
```

环境变量（可选）：

- `POLAR_MAX_FRAMES` — 每 SNR 点最大帧数（默认 100000）
- `POLAR_MIN_ERRORS` — 最少错误帧数（默认 100）

## 说明

- LLR 约定：`LLR = ln P(y|0)/P(y|1) = 2y/σ²`。
- 编码器与递归 SC 译码器配对；`sc_decode()` 默认使用递归实现。
- 设计信噪比 `design_Eb/N0=2.5 dB` 下，(256,128) 码约在 **7 dB 以上** 才出现明显 BLER 下降，仿真 Eb/N0 范围已设为 2–10 dB 以便观察曲线。
