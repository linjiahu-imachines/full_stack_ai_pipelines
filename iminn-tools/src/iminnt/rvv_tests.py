
from typing import Dict
from pathlib import Path
from .constants import IMI_SDK_ROOT, DEV_ENV_ROOT, IMI_RISCV_ARCH, CROSS_TOOLCHAIN_PATH, IMI_ENV, USE_DEBUG_SYMS
from .registry import register_class
from .core import BenchSimRunner

STRIP_BIN=False
BUILD_DEBUG=True

@register_class("rvv-tests")
class RISCVTest(BenchSimRunner):

    @property
    def env(self) -> Dict:
        return IMI_ENV

    @property
    def use_llvm_src(self) -> bool:
        return True

    @property
    def debug_cmake(self) -> bool:
        return False

    @property
    def clang_root(self) -> Path:
        return IMI_SDK_ROOT

    @property
    def remote_path(self):
        return str(DEV_ENV_ROOT / "rvv_tests")

    @property
    def target(self) -> str:
        return "rvv-tests"

    @property
    def default_bin(self) -> str:
        return "rvv_test_runner"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "rvv_tests"

    @property
    def install_dir(self) -> Path:
        return self.root / "rvv-tests-build"

    @property
    def build_dir(self) -> Path:
        return self.root / "rvv-tests-build"

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        cmake_args = {
            "CMAKE_BUILD_TYPE": "Debug",
            "CMAKE_INSTALL_PREFIX": f"{self.install_dir}",
            "CROSS_STATIC": "ON",
            "CMAKE_TOOLCHAIN_FILE": f"{CROSS_TOOLCHAIN_PATH}",
        }
        if STRIP_BIN:
            cmake_args["BINSTRIP"] = "ON"

        if BUILD_DEBUG:
            cmake_args["CMAKE_BUILD_TYPE"] = "Debug"

        if USE_DEBUG_SYMS:
            cmake_args["DEBUG_SYMS"] = "ON"
        
        cmake_arg_str = " ".join([f"-D{k}={v}" for k, v in cmake_args.items()])
        cmd = []
        cmd.append("rm -rf rvv-tests-build")
        cmd.append(f"cmake -G Ninja -S {self.root} -B {self.build_dir} {cmake_arg_str}")
        cmd.append(f"cmake --build {self.build_dir} -j{threads}")
        return cmd

    @property
    def rebuild_cmd(self) -> list:
        if self.debug_cmake:
            return [f"cmake --build {self.build_dir} --verbose"]
        else:
            return [f"cmake --build {self.build_dir}"]

    @property
    def default_runs(self) -> Dict:
        commands = {
            "mm_1x16x4_v1": {"bin": f"{self.default_bin}", "args": "-k 0"},
            "mm_4x16x4_v1": {"bin": f"{self.default_bin}", "args": "-k 1"},
            "mm_8x16x4_v1": {"bin": f"{self.default_bin}", "args": "-k 2"},
            "mm_2x16x4_v2": {"bin": f"{self.default_bin}", "args": "-k 3"},
            "mm_4x16x4_v2": {"bin": f"{self.default_bin}", "args": "-k 4"},
            "mm_8x16x4_v2": {"bin": f"{self.default_bin}", "args": "-k 5"},
            "mm_12x16x4_v2": {"bin": f"{self.default_bin}", "args": "-k 6"},
            "mm_16x16x4_v2": {"bin": f"{self.default_bin}", "args": "-k 7"},
            "mm_1x8x4_v1": {"bin": f"{self.default_bin}", "args": "-k 8"},
            "mm_2x8x4_v2": {"bin": f"{self.default_bin}", "args": "-k 9"},
            "mm_1x16x4_v1_micro": {"bin": f"{self.default_bin}", "args": "-k 10"},
            "mm_1x16x8_v1": {"bin": f"{self.default_bin}", "args": "-k 11"},
            "mm_1x32x4_v1": {"bin": f"{self.default_bin}", "args": "-k 12"},
            "mm_1x64x4_v1": {"bin": f"{self.default_bin}", "args": "-k 13"},
            "mm_1x4x16_rvv": {"bin": f"{self.default_bin}", "args": "-k 14"},
            "mm_1x4x16_rvv_asm": {"bin": f"{self.default_bin}", "args": "-k 15"}
        }
        return commands
