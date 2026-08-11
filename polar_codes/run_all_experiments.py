#!/usr/bin/env python3
"""顺序运行全部实验（单进程）"""
import os
import subprocess
import sys

os.chdir(os.path.dirname(__file__))
os.makedirs("results", exist_ok=True)

for script in ["run_exp1.py", "run_exp2.py", "run_exp3.py"]:
    log = f"results/{script.replace('.py', '')}.log"
    print(f"=== Running {script} ===", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run([sys.executable, script], stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"{script} failed with code {proc.returncode}", flush=True)
        sys.exit(proc.returncode)

with open("results/status.txt", "w", encoding="utf-8") as f:
    f.write("ALL_DONE\n")
print("ALL_DONE", flush=True)
