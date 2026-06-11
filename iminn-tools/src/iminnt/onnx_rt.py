from pathlib import Path
from typing import List, Dict, Optional
from .constants import IMI_SDK_ROOT, DEV_ENV_ROOT, IMI_RISCV_ARCH, CROSS_TOOLCHAIN_PATH, IMI_ENV, USE_DEBUG_SYMS, QEMU_USER_BIN, IMINNT_ROOT, AMX_REMOTE, AMX_ENV
from .registry import register_class
from .core import BenchSimRunner

PATCH_FILE = "ort_rvv_build_fixes_v2.patch"


class ONNXRTBase(BenchSimRunner):
    def __init__(self, 
                target: str,
                build_dir: str, 
                install_dir: str,
                xnnpack_build_dir: str,
                env: Optional[Dict] = None, 
                remote_info: Optional[Dict] = None, 
                use_qemu: bool = True):
        self._build_dir = build_dir
        self._install_dir = install_dir

        self._xnnpack_build_dir = xnnpack_build_dir
        self._env = env or {}
        self._target = target
        self._remote_info = remote_info
        self._use_qemu = use_qemu

    @property
    def custom_scripts(self) -> List[str]:
        return [PATCH_FILE]

    @property
    def shell_build(self) -> bool:
        return True
    
    @property
    def extra_build_cmds(self) -> List[str]:
        return []

    @property
    def git_tag(self) -> Optional[str]:
        return "v1.22.2"

    @property
    def post_init(self) -> List[str]:
        return [
            f"mkdir -p {self.root}",
            f"git -C {self.root} apply {PATCH_FILE}"
        ]

    @property
    def remote_path(self):
        return "https://github.com/microsoft/onnxruntime"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "onnxruntime"

    @property
    def target(self) -> str:
        return self._target

    @property
    def env(self) -> Dict:
        return self._env

    @property
    def is_riscv(self) -> bool:
        return self.target == "onnxrt_imi" or self.target == "onnxrt_rvv"
    
    @property
    def is_imi(self) -> bool:
        return self.target == "onnxrt_imi"
    
    @property
    def is_aarch64(self) -> bool:
        return self.target == "onnxrt_neoverse"

    @property
    def xnnpack_root(self) -> Path:
        return (DEV_ENV_ROOT / "XNNPACK")

    def check_deps(self, is_build: bool):
        if not self.xnnpack_root.exists():
            raise RuntimeError(f"ONNXRT requires XNNPACK dependency to be initialized. Please initialize with `iminnt -t xnnpack_imi`, then try again.")

    @property
    def xnnpack_build_dir(self) -> Path:
        return self.xnnpack_root / self._xnnpack_build_dir

    @property
    def build_args(self) -> Dict:
        cfg_args = {
            "--config": "RelWithDebInfo",
            "--build_dir": f"{self.build_dir}",
            "--skip_submodule_sync": None,
            "--parallel": None,
            "--skip_tests": None,
            "--compile_no_warning_as_error": None,
            "--use_xnnpack": None,
            # "--build_micro_benchmarks": None
        }
        cmake_args = {
            "CMAKE_INSTALL_PREFIX": f"{self.install_dir}",
            "CMAKE_VERBOSE_MAKEFILE": "ON"
        }

        if self.is_riscv or self.is_aarch64:
            arch_cmake_args = {
                "CROSS_STATIC": "ON",
                "CMAKE_TOOLCHAIN_FILE": f"{CROSS_TOOLCHAIN_PATH}",
                "HAVE_POSIX_REGEX": "0",
                "HAVE_STEADY_CLOCK": "0",
                "HAVE_STD_REGEX": "0",
                "BUILD_SHARED_LIBS": "OFF",
                "onnxruntime_CROSS_COMPILING": "ON",
                "onnxruntime_BUILD_UNIT_TESTS": "ON",
                "onnxruntime_BUILD_BENCHMARKS": "ON"
            }
            if self.is_riscv:
                arch_cmake_args["LIBM"] = f'{IMI_SDK_ROOT}/sysroot/usr/lib/libm.a'
            if self.is_imi:
                arch_cmake_args["XNN_ENABLE_RISCV_IMI"] = "ON"
            elif self.is_riscv:
                arch_cmake_args["XNN_ENABLE_RISCV_VECTOR"] = "1"
                arch_cmake_args["XNN_ENABLE_RISCV_IMI"] = "0"
        else:
            arch_cmake_args = {
                "BUILD_SHARED_LIBS": "OFF",
                "onnxruntime_BUILD_UNIT_TESTS": "ON",
                "onnxruntime_BUILD_BENCHMARKS": "ON"
            }

        arch_cmake_args["FETCHCONTENT_SOURCE_DIR_GOOGLEXNNPACK"] = str(self.xnnpack_root)
        cmake_args.update(arch_cmake_args)
        args = []
        for k, v in cfg_args.items():
            if v is None:
                args.append(k)
            else:
                args.append(f"{k}={v}")
        cmake_extras = " --cmake_extra_defines " + " ".join([f"{k}={v}" for k,v in cmake_args.items()])
        return " ".join(args) + cmake_extras

    @property
    def install_dir(self) -> Path:
        return self.root / self._install_dir

    @property
    def build_dir(self) -> Path:
        return self.root / self._build_dir

    def get_build_cmd(self, threads=None) -> list:
        cmds = [ 
                f"rm -rf {self.build_dir}", 
                f"rm -rf {self.install_dir}",
                f"python3 {self.root}/tools/ci_build/build.py {self.build_args}",
                f"python3 {self.root}/tools/ci_build/build.py {self.build_args} --target install",
            ]
        return cmds

    @property
    def rebuild_cmd(self) -> list:
        return [
            f"python3 {self.root}/tools/ci_build/build.py {self.build_args} --build",
            f"python3 {self.root}/tools/ci_build/build.py {self.build_args} --target install",
        ]

    @property
    def test_bin(self) -> str:
        return f"{self.build_dir}/tools/benchmark/benchmark_model"
    
    @property
    def default_bin(self) -> str:
        return self.test_bin

    @property
    def use_qemu(self) -> bool:
        return self._use_qemu

    @property
    def remote_info(self) -> Optional[Dict]:
        return self._remote_info
    
    @property
    def bench_bin(self) -> str:
        return f"{self.build_dir}/RelWithDebInfo/onnxruntime_perf_test"
    
    @property
    def test_bin(self) -> str:
        return f"{self.build_dir}/RelWithDebInfo/onnx_test_runner"

    @property
    def bench_args(self) -> str:
        bargs = []
        bargs.append("-m times")
        bargs.append("-e xnnpack")
        bargs.append("-I")
        bargs.append("-r 1")
        bargs.append("-x 1")
        bargs.append("-y 1")
        bargs.append("-o 1")
        bargs.append("-v")
        return " ".join(bargs)

    @property
    def default_runs(self) -> Dict:
        return {
            "benchtest": {"bin": self.bench_bin, "args": "--help"},
            "onnxtest": {"bin": self.test_bin, "args": "--help"},
            # "model_test": {"bin": self.bench_bin, "args": f"{self.bench_args} {self.root}/minilm_l6_v2_fp32.onnx"},
            # "model_test": {"bin": self.bench_bin, "args": f"{self.bench_args} {self.root}/minilm_l6_v2_int8.onnx"},
            # "model_test": {"bin": self.bench_bin, "args": f"{self.bench_args} {self.root}/bert_int8.onnx"},
            "model_test": {"bin": self.bench_bin, "args": f"{self.bench_args} {IMINNT_ROOT}/scratch/minilm.onnx"},
            "model_test_working": {"bin": self.bench_bin, "args": f"{self.bench_args} {IMINNT_ROOT}/scratch/tiny-llama.onnx"},
            "model_test_working0": {"bin": self.bench_bin, "args": f"{self.bench_args} {IMINNT_ROOT}/scratch/tiny-phi3.onnx"},
            "model_test_working1": {"bin": self.bench_bin, "args": f"{self.bench_args} {IMINNT_ROOT}/scratch/fashion-clip.onnx"},
            "model_test_working2": {"bin": self.bench_bin, "args": f"{self.bench_args} {IMINNT_ROOT}/scratch/answerai-colbert-int8.onnx"},
            "model_test_help": {"bin": self.bench_bin, "args": f"--help"},
        }


@register_class("onnx_x86")
class ONNXRTX86(ONNXRTBase):
    def __init__(self):
        super().__init__(
            "onnxrt_x86",
            "linux-build-x86",
            "onnxrt-x86-install",
            "linux-build-x86",
            use_qemu=False
        )

@register_class("onnx_amx")
class ONNXRTAMX(ONNXRTBase):
    def __init__(self):
        super().__init__(
            "onnx_amx",
            "linux-build-amx",
            "onnxrt-amx-install",
            "linux-build-amx",
            use_qemu=False,
            remote_info=AMX_REMOTE,
            env=AMX_ENV
        )

@register_class("onnx_imi")
class ONNXRTIMI(ONNXRTBase):
    def __init__(self):
        super().__init__(
            "onnxrt_imi",
            "linux-build-imi",
            "onnxrt-imi-install",
            "linux-build-imi",
            env=IMI_ENV
        )