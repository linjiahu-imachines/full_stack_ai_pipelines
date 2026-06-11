from pathlib import Path
from typing import List, Dict, Optional
from .utils import shell
from .constants import IMI_SDK_ROOT, DEV_ENV_ROOT, IMI_RISCV_ARCH, RVV_RISCV_ARCH, CROSS_TOOLCHAIN_PATH, IMI_ENV, RVV_ENV, \
                        USE_DEBUG_SYMS, PROMPTS_DIR, AMX_REMOTE, AMX_ENV, RISCV_ENV_ROOT
from .registry import register_class
from .core import BenchSimRunner

class IREEBase(BenchSimRunner):
    def __init__(self, 
                target: str,
                build_dir: str, 
                install_dir: str, 
                env: Optional[Dict] = None, 
                remote_info: Optional[Dict] = None, 
                use_qemu: bool = True,
                build_ref: bool = True,
                features: Optional[list] = None
                ):
        self._build_dir = build_dir
        self._install_dir = install_dir
        self.build_ref = build_ref
        self._env = env or {}
        self._target = target
        self._remote_info = remote_info
        self._use_qemu = use_qemu
        self.features = features or []

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/iree"

    @property
    def custom_scripts(self) -> List[str]:
        return ["gtest_fix.patch"]

    @property
    def post_init(self) -> List[str]:
        cmds = [
            f"git -C {self.root} remote add upstream https://github.com/iree-org/iree",
            f"git -C {self.root} checkout upstream",
            f"git -C {self.root} checkout master",
            f"mkdir -p {self.root}/artifacts",
            f"git -C {self.root}/third_party/googletest apply {self.root}/gtest_fix.patch",
            f"mv {self.root}/third_party/llvm-project {self.root}/third_party/iree-llvm-project",
            f"ln -s {RISCV_ENV_ROOT}/llvm-project {self.root}/third_party/llvm-project",
            f"wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/mobilenet/model/mobilenetv2-10.onnx -P {self.root}/artifacts"
        ]
        return cmds

    @property
    def upstream_remote(self) -> Optional[str]:
        return "https://github.com/iree-org/iree"

    @property
    def target(self) -> str:
        return self._target

    @property
    def env(self) -> Dict:
        return self._env

    @property
    def is_riscv(self) -> bool:
        return False
    
    @property
    def is_imi(self) -> bool:
        return False

    @property
    def is_x86(self) -> bool:
        return False
    
    @property
    def is_aarch64(self) -> bool:
        return False

    @property
    def install_dir(self) -> Path:
        return self.root / self._install_dir

    @property
    def build_dir(self) -> Path:
        return self.root / self._build_dir

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "iree"

    @property
    def host_build_dir(self) -> Path:
        return self.root / "linux-build-x86"

    @property
    def host_install_dir(self) -> Path:
        return self.root / "iree-x86-install"

    @property
    def artifacts_dir(self):
        return f"{self.root}/artifacts"

    def check_deps(self, is_build: bool):
        if is_build and self.target != "iree_x86" and not self.host_build_dir.exists():
            raise RuntimeError(f"Cross-compiling IREE requires an x86 (host) build first for host-tool usage. Please build with `iminnt -t iree_x86 build`, then try again.")
        elif not (RISCV_ENV_ROOT / "llvm-project").exists():
            raise RuntimeError(f"IREE requires llvm dependency to be initialized. Please initialize with `iminnt -t llvm`, then try again.")

    def has_ccache(self):
        return shell(["which", "ccache"], no_fail=True, silent=True)["exit_code"] == 0

    @property
    def cmake_args(self):
        BUILD_TESTS=True
        base_args = {
            "CMAKE_BUILD_TYPE": "RelWithDebInfo",
            "CMAKE_INSTALL_PREFIX": f"{self.install_dir}",
            "IREE_BUILD_BINDINGS_TFLITE": "OFF",
            "IREE_BUILD_BINDINGS_TFLITE_JAVA": "OFF",
        }
        if self.has_ccache():
            base_args["CMAKE_CXX_COMPILER_LAUNCHER"] = "ccache"
            base_args["CMAKE_C_COMPILER_LAUNCHER"] = "ccache"

        if self.is_riscv:
            
            base_args["CROSS_STATIC"] = "ON"
            base_args["CMAKE_TOOLCHAIN_FILE"] = f"{CROSS_TOOLCHAIN_PATH}"
            base_args["IREE_HOST_BIN_DIR"] = f"{self.host_install_dir}/bin"
            base_args["RISCV_CPU"] = f"linux-riscv_64"
            base_args["IREE_BUILD_COMPILER"] = "OFF"
            base_args["RISCV_TOOLCHAIN_ROOT"] = f"{IMI_SDK_ROOT}"
            base_args["IREE_ENABLE_CPUINFO"] = "OFF"
            base_args["BUILD_SHARED_LIBS"] = "OFF"
            base_args["IREE_BUILD_TESTS"] = "OFF"
            base_args["IREE_BUILD_SAMPLES"] = "OFF"
            base_args["IREE_BUILD_TESTS"] = "ON"
            base_args["IREE_VULKAN_DISABLE"] = "1"
            # Currently, all multithreading is disabled, so we disable in the build as well.
            # The options below are all an effort to successfully build tests with threading disabled
            # base_args["IREE_ENABLE_THREADING"] = "OFF"
            # base_args["IREE_HAL_DRIVER_DEFAULTS"] = "OFF"
            # base_args["IREE_HAL_DRIVER_LOCAL_SYNC"] = "ON"
            # base_args["IREE_HAL_EXECUTABLE_LOADER_DEFAULTS"] = "OFF"
            # base_args["IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF"] = "ON"
            # base_args["IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE"] = "ON"
            # base_args["IREE_SYNCHRONIZATION_DISABLE_UNSAFE"] = "1"

        elif self.is_aarch64:
            raise NotImplementedError(f"Aarch64 not yet implemented")
        else:
            assert self.is_x86
            # clang unfortunately is not able to build LLVM because of all the translation units, so we have to use gcc/g++ ere.
            # base_args["CMAKE_C_COMPILER"] = "clang"
            # base_args["CMAKE_CXX_COMPILER"] = "clang++"
            base_args["CMAKE_C_COMPILER"] = "gcc"
            base_args["CMAKE_CXX_COMPILER"] = "g++"
            base_args["IREE_ENABLE_LLD"] = "ON"
            base_args["IREE_ENABLE_SPLIT_DWARF"] = "ON"
            base_args["IREE_ENABLE_THIN_ARCHIVES"] = "ON"
            base_args["IREE_BUILD_SAMPLES"] = "OFF"
            base_args["IREE_BUILD_TESTS"] = "ON"
            base_args["LLVM_BUILD_TESTS"] = "OFF"
            base_args["LLVM_TARGETS_TO_BUILD"] = "X86;AArch64;RISCV"
        return base_args

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        cmake_args = " ".join([f"-D{k}={v}" for k,v in self.cmake_args.items()])
        cmds = [f"rm -rf {self.build_dir}", f"rm -rf {self.install_dir}"]
        cmds = cmds + [
            f"cmake -G Ninja -S {self.root} -B {self.build_dir} {cmake_args}",
            f"cmake --build {self.build_dir} -j{threads} -t install"
        ]
        return cmds

    @property
    def rebuild_cmd(self) -> list:
        return [f"cmake --build {self.build_dir} -j -t install"]

    @property
    def test_bin(self) -> str:
        return f"{self.build_dir}/bin/test-backend-ops"
    
    @property
    def default_bin(self) -> str:
        return f"{self.install_dir}/bin/iree-run-module"

    @property
    def bench_bin(self) -> str:
        return f"{self.install_dir}/bin/iree-benchmark-module"

    @property
    def use_qemu(self) -> bool:
        return self._use_qemu

    @property
    def remote_info(self) -> Optional[Dict]:
        return self._remote_info

    
    @property
    def host_compile_bin(self):
        return f"{self.host_install_dir}/bin/iree-compile"

    @property
    def host_dump_bin(self):
        return f"{self.host_install_dir}/bin/iree-dump-module"

    def get_iree_compile_args(self, arch: str) -> Dict:
        args = {}
        args["--iree-hal-local-target-device-backends"] = "llvm-cpu"
        # this disables threading
        args["--iree-opt-level"] = "O2"
        args["--iree-opt-data-tiling"] = None
        args["--iree-llvmcpu-enable-ukernels"] = "all"

        if arch == "rvv" or arch == "imi":
            args["--iree-llvmcpu-target-triple"] = "riscv64"
            args["--iree-llvmcpu-target-abi"] = "lp64d"
            if arch == "imi":
                arch_spec = f"{IMI_RISCV_ARCH.lower()}"
            else:
                arch_spec = f"{RVV_RISCV_ARCH.lower()}"
            std_feats = ",".join([f"+{f}" for f in arch_spec.split("_")[0].replace("rv64", "").replace("g", "mafd")])

            arch_spec = std_feats + ",+" + ",+".join(arch_spec.split("_")[1:])
            args["--iree-llvmcpu-target-cpu-features"] = arch_spec
            args["--riscv-v-fixed-length-vector-lmul-max"] = "8"
        elif arch == "aarch64":
            raise NotImplementedError(f"AArch64 not yet supported")
        else:
            assert arch == "x86"
            args["--iree-llvmcpu-target"] = "host"
        # Ensure single-threaded execution
        # args["--iree-execution-model"] = "host-only" 
        args["--mlir-disable-threading"] = None 
        args["--iree-llvmcpu-disable-distribution"] = None
        args["--iree-llvmcpu-number-of-threads"] = "1"
        return args

    @property
    def arch_id(self):
        return self.target.split("_")[1]

    @property
    def default_runs(self) -> Dict:
        DO_DEBUG = True
        # DEBUG_ARGS = "--mlir-print-ir-after=iree-codegen-cpu-lower-to-ukernels"
        defaults = {}
        models = {}
        models["mobilenetv2"] = {"--function": "torch-jit-export", "--input": "\"1x3x224x224xf32=0\""}
        models["large_linalg_matmul"] = {"path": f"{self.root}/tests/e2e/linalg", "--function": "matmul_2048x512x1024_f32_f32"}
        mm_benchmarks = ["dyn_matmul_f16_f16_f16", "dyn_matmul_i8_i8_i32", "dyn_matmul_i8_i4_i32"]
        targets = ["rvv", "imi", "x86"]
        iree_rvv_compile_args = " ".join([f"{k}={v}" if v else k for k, v in self.get_iree_compile_args('rvv').items()])
        iree_imi_compile_args = " ".join([f"{k}={v}" if v else k for k, v in self.get_iree_compile_args('imi').items()])
        iree_x86_compile_args = " ".join([f"{k}={v}" if v else k for k, v in self.get_iree_compile_args('x86').items()])
        for t in targets:
            iree_compile_args = " ".join([f"{k}={v}" if v else k for k, v in self.get_iree_compile_args(t).items()])
            for m, margs in models.items():
                # For each backend, add compile option. We have to compile using the host target, so we need to add distinct compile commands for each
                mpath = margs.get("path", f"{self.root}/artifacts")
                # This compiles and directly generates a vm-bytecode file
                defaults[f"{m}_compile_{t}"] = {
                    "bin": self.host_compile_bin,
                    "args": f"--iree-hal-target-device=local {iree_compile_args} \
                        {mpath}/{m}.mlir -o {self.artifacts_dir}/{m}_{t}.vmfb"
                }
                # This compiles and generates a c file
                defaults[f"{m}_compile_{t}_c"] = {
                    "bin": self.host_compile_bin,
                    "args": f"--iree-hal-target-device=local {iree_compile_args} \
                        {mpath}/{m}.mlir --iree-vm-c-module-optimize --output-format=vm-c -o {self.artifacts_dir}/{m}_{t}.c"
                }
                # This compiles and generates an assembly file
                defaults[f"{m}_compile_{t}_asm"] = {
                    "bin": self.host_compile_bin,
                    "args": f"--iree-hal-target-device=local {iree_compile_args} \
                        {mpath}/{m}.mlir --output-format=vm-asm -o {self.artifacts_dir}/{m}_{t}.S"
                }
                ## Add execution command
                if self.arch_id == t:
                    marg_str = " ".join([f"{k}={v}" for k, v in margs.items() if k != "path"])
                    defaults[f"{m}_run"] = {
                        "bin": self.default_bin,
                        "args": f"--device=local-task --module={self.artifacts_dir}/{m}_{self.arch_id}.vmfb {marg_str}"
                    }
            for mm_bench in mm_benchmarks:
                fpath = Path(self.root) / "benchmarks" / f"{mm_bench}.mlir"
                if DO_DEBUG:
                    debug_path = f"{self.artifacts_dir}/{mm_bench}_{t}"
                    compile_args = f"{iree_compile_args} --iree-hal-dump-executable-files-to={debug_path}"
                    # compile_args = f"{iree_compile_args} --dump-compilation-phases-to={debug_path}"
                else:
                    compile_args = iree_compile_args
                defaults[f"{mm_bench}_compile_{t}"] = {
                    "bin": self.host_compile_bin,
                    "args": f"--iree-hal-target-device=local {compile_args} \
                        {fpath} -o {self.artifacts_dir}/{mm_bench}_{t}.vmfb"
                }
                defaults[f"{mm_bench}_compile_{t}_c"] = {
                    "bin": self.host_compile_bin,
                    "args": f"--iree-hal-target-device=local {compile_args} \
                        {fpath} --iree-vm-c-module-optimize --output-format=vm-c -o {self.artifacts_dir}/{mm_bench}_{t}.c"
                }

                defaults[f"{mm_bench}_dump_{t}"] = {
                    "bin": self.host_dump_bin,
                    "args": f"{self.artifacts_dir}/{mm_bench}_{t}.vmfb"
                }

                if self.arch_id == t:
                    MM_DIM = 128
                    operand_types = mm_bench.split("_")[2:]
                    operand_args = " ".join([f"--input={MM_DIM}x{MM_DIM}x{ot}" for ot in operand_types])
                    single_thread_args = "--task_topology_group_count=1 --task_topology_max_group_count=1 --task_topology_cpu_ids=0"
                    mm_bench_args = f"--function=matmul_dynamic --benchmark_repetitions=1 --benchmark_min_warmup_time=0 --benchmark_enable_random_interleaving=false {operand_args} {single_thread_args}"
                    defaults[f"{mm_bench}_run"] = {
                        "bin": self.bench_bin,
                        "args": f"--device=local-sync --module={self.artifacts_dir}/{mm_bench}_{t}.vmfb {mm_bench_args}"
                    }
        # Any custom ones here
        defaults["compile_help"] = {"bin": self.host_compile_bin, "args": "--help"}
        defaults["ukernel_mm_test"] = {"bin": f"{self.build_dir}/runtime/src/iree/builtins/ukernel/tools/mmt4d_test", "args": ""}
        return defaults


@register_class("iree_x86")
class IREEX86(IREEBase):
    def __init__(self):
        super().__init__(
            "iree_x86",
            "linux-build-x86",
            "iree-x86-install",
            use_qemu=False
        )

    @property
    def is_x86(self) -> bool:
        return True


@register_class("iree_imi")
class IREEIMI(IREEBase):
    def __init__(self):
        super().__init__(
            "iree_imi",
            "linux-build-imi",
            "iree-imi-install",
            env=IMI_ENV
        )

    @property
    def is_riscv(self) -> bool:
        return True
    
    @property
    def is_imi(self) -> bool:
        return True

    @classmethod
    def clang_root(cls) -> Path:
        return IMI_SDK_ROOT

@register_class("iree_rvv")
class IREERVV(IREEBase):
    def __init__(self):
        super().__init__(
            "iree_rvv",
            "linux-build-rvv",
            "iree-rvv-install",
            env=IMI_ENV
        )

    @property
    def is_riscv(self) -> bool:
        return True

    @classmethod
    def clang_root(cls) -> Path:
        return IMI_SDK_ROOT
