# 极化码编译码仿真

纯 NumPy/SciPy 实现的极化码（Polar Codes）仿真框架，包含 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛 BLER 评估。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python validate.py          # 单元测试
python run_exp1.py          # SC 仿真
python run_exp2.py          # SCL / CA-SCL
python run_exp3.py          # BP 对比
```

快速冒烟测试（减少帧数）：

```bash
POLAR_QUICK=1 python run_exp1.py
```

结果输出至 `results/`。
