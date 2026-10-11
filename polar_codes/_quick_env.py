"""POLAR_QUICK=1 时缩短蒙特卡洛仿真规模（供 cron / 冒烟测试）"""
import os


def apply_quick_defaults():
    if os.environ.get("POLAR_QUICK") != "1":
        return
    os.environ.setdefault("POLAR_MAX_FRAMES", "3000")
    os.environ.setdefault("POLAR_MIN_ERRORS", "15")
    os.environ.setdefault("POLAR_EB_MIN", "1.0")
    os.environ.setdefault("POLAR_EB_MAX", "4.5")
    os.environ.setdefault("POLAR_EB_STEP", "0.5")
    os.environ.setdefault("POLAR_SKIP_N1024", "1")
