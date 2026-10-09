# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
python validate.py
```

## 实验

```bash
python run_exp1.py   # SC
python run_exp2.py   # SCL / CA-SCL
python run_exp3.py   # BP 对比
```

可选环境变量：`POLAR_MAX_FRAMES`、`POLAR_MIN_ERRORS`、`POLAR_EB_MIN`、`POLAR_EB_MAX`、`POLAR_EB_STEP`、`POLAR_SKIP_N1024=1`。

输出目录：`results/`。
