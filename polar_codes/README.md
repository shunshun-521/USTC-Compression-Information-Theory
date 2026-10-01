# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 快速开始

```bash
pip install -r requirements.txt
python3 validate.py
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

快速冒烟（较少帧数、较粗 SNR 步进）：

```bash
POLAR_FAST_SIM=1 python3 run_exp1.py
POLAR_FAST_SIM=1 python3 run_exp2.py
POLAR_FAST_SIM=1 python3 run_exp3.py
```

结果输出在 `results/`。

## 说明

- SC 译码采用对数域 f 函数与比特倒序相位更新（与标准极化蝶形编码一致）。
- `f_operation` 对外保留 min-sum 接口；BP 译码使用 min-sum 因子图迭代。
- 编码器单元测试向量：`u=[1,0,1,1] -> x=[1,1,0,1]`（与分阶段蝶形 XOR 一致）。
