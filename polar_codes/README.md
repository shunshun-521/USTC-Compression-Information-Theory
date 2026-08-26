# 极化码编译码仿真

纯 Python（NumPy/SciPy）实现的极化码编译码仿真，包含 GA 构造、SC/SCL/BP 译码与蒙特卡洛仿真。

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python3 validate.py          # 数值校验
python3 construction.py      # 打印 GA 构造验证信息
```

## 运行实验

```bash
# 快速仿真（自动化/调试）
POLAR_FAST_SIM=1 python3 run_exp1.py
POLAR_FAST_SIM=1 python3 run_exp2.py
POLAR_FAST_SIM=1 python3 run_exp3.py

# 完整仿真（耗时较长）
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

## 目录结构

- `construction.py` — 高斯近似（GA）极化码构造
- `encoder.py` — 蝶形编码 + 比特倒序
- `decoder_sc.py` — PSCD 串行抵消译码
- `decoder_scl.py` — SCL / CA-SCL 译码
- `decoder_bp.py` — BP 译码（min-sum + 早停）
- `channel.py` — BPSK-AWGN 信道
- `simulation.py` — 蒙特卡洛仿真主循环
- `utils.py` — CRC、绘图、结果保存
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 三组实验
- `results/` — CSV 与图像输出

## 说明

- LLR 约定：`LLR = ln P(y|0)/P(y|1) = 2y/σ²`
- 编码采用 `G_N = B_N F^{⊗n}`；译码前对信道 LLR 做比特倒序以匹配 PSCD
- 完整仿真建议每个信噪比点 `max_frames=100000`、`min_errors=100`
