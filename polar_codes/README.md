# Polar Codes Simulation

极化码（Polar Codes）编译码仿真框架，包含 GA 构造、SC/SCL/CA-SCL/BP 译码及蒙特卡洛仿真。

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
├── run_exp1.py           # 实验一：SC
├── run_exp2.py           # 实验二：SCL
├── run_exp3.py           # 实验三：BP
└── results/              # 输出结果
```

## 依赖

```bash
pip install numpy scipy matplotlib
```

## 运行实验

```bash
cd polar_codes/
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

## 单元测试

```bash
python validate.py
```
