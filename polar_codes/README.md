# Polar Codes Simulation

极化码（Polar Codes）编译码仿真全流程，包含 GA 构造、SC/SCL/CA-SCL/BP 译码及蒙特卡洛仿真。

## 目录结构

```
polar_codes/
├── construction.py       # GA 极化码构造
├── encoder.py            # 编码器
├── decoder_sc.py         # SC 译码器
├── decoder_scl.py        # SCL / CA-SCL 译码器
├── decoder_bp.py         # BP 译码器
├── channel.py            # AWGN + BPSK
├── simulation.py         # 蒙特卡洛仿真
├── utils.py              # 工具函数
├── validate.py           # 单元测试
├── run_exp1.py           # SC 仿真
├── run_exp2.py           # SCL 仿真
├── run_exp3.py           # BP 仿真
└── results/              # 输出结果
```

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python validate.py          # 运行单元测试
python run_exp1.py          # SC 仿真
python run_exp2.py          # SCL 仿真
python run_exp3.py          # BP 仿真
```

快速仿真模式（减少帧数）：

```bash
POLAR_FAST_SIM=1 python run_exp1.py
```

## 依赖

- Python 3.8+
- NumPy, SciPy, Matplotlib

不使用任何极化码第三方库。
