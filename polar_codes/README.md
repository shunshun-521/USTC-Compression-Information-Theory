# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 安装

```bash
pip install -r requirements.txt
```

## 校验与实验

```bash
cd polar_codes
python3 verify.py
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

快速冒烟（较少帧数）：

```bash
POLAR_QUICK=1 python3 run_exp1.py
```

结果输出至 `results/`。
