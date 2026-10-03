"""与 validate.py 相同，供实验脚本与 CI 调用的校验入口。"""
from validate import *  # noqa: F401, F403

if __name__ == "__main__":
    import validate

    validate.test_encoder()
    validate.test_construction()
    validate.test_crc()
    validate.test_sc_lossless()
    validate.test_sc_recursive_match()
    validate.test_scl_equiv_sc()
    print("\n所有校验通过。")
