# Polar Codes Simulation

极化码编译码仿真：GA 构造、SC/SCL/CA-SCL/BP 译码、蒙特卡洛仿真。

## 运行

```bash
cd polar_codes
python run_exp1.py   # SC 仿真
python run_exp2.py   # SCL / CA-SCL 仿真
python run_exp3.py   # BP 仿真
```

可选环境变量加速测试：

```bash
POLAR_MAX_FRAMES=1000 POLAR_MIN_ERRORS=20 python run_exp1.py
```
