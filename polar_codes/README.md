# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/BP 译码与蒙特卡洛仿真，无第三方极化码库。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py   # 打印 GA 验证索引
python run_exp1.py       # SC 仿真
python run_exp2.py       # SCL / CA-SCL
python run_exp3.py       # SC / SCL / BP 对比
```

结果输出至 `results/`。

## 说明

- 编码采用与 5G/Sionna 一致的 XOR 蝶形结构；SC 译码在 g 节点使用部分重编码比特（`u_hat_up`）。
- 蒙特卡洛仿真使用 `ga_construction_for_simulation()`：在 3GPP 子信道排序上选取信息位，与译码器匹配；`ga_construction()` 仍用于 `frozen_sets.txt` 报告。
