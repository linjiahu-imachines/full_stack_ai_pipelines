from pathlib import Path
from typing import List, Dict, Optional
from .constants import IMI_SDK_ROOT, DEV_ENV_ROOT, RVV_ENV, IMI_RISCV_ARCH, \
                     CROSS_TOOLCHAIN_PATH, IMI_ENV, USE_DEBUG_SYMS, ORYON_ENV, ORYON_REMOTE
from .registry import register_class
from .core import BenchSimRunner

USE_THIRDPARTY_TFLITE=False
GIT_TAG = "v2.20.0"

class LiteRTBase(BenchSimRunner):
    def __init__(self, 
                target: str,
                build_dir: str, 
                install_dir: str,
                xnnpack_build_dir: str,
                env: Optional[Dict] = None, 
                remote_info: Optional[Dict] = None, 
                use_qemu: bool = True):
        if USE_THIRDPARTY_TFLITE:
            self._build_dir = f"{build_dir}-lrt"
            self._install_dir = f"{install_dir}-lrt"
        else:
            self._build_dir = build_dir
            self._install_dir = install_dir

        self._xnnpack_build_dir = xnnpack_build_dir
        self._env = env or {}
        self._target = target
        self._remote_info = remote_info
        self._use_qemu = use_qemu

    @property
    def custom_scripts(self) -> List[str]:
        return ["tflite_benchmark_roi.patch"]

    @property
    def shell_build(self) -> bool:
        return True
    
    @property
    def extra_build_cmds(self) -> List[str]:
        return []

    @property
    def git_tag(self) -> Optional[str]:
        return GIT_TAG
    
    @property
    def post_init(self) -> List[str]:
        return [
            f"mkdir -p {self.root}",
            f"wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz",
            f"tar -xzvf {self.root}/mobilenet_v1_1.0_224.tgz",
            f"git -C {self.root} apply tflite_benchmark_roi.patch"
        ]

    @property
    def remote_path(self):
        return "https://github.com/tensorflow/tensorflow"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "tensorflow"

    @property
    def tflite_path(self):
        return self.root / "tensorflow" / "lite"

    @property
    def target(self) -> str:
        return self._target

    @property
    def env(self) -> Dict:
        return self._env

    @property
    def is_riscv(self) -> bool:
        return self.target == "litert_imi" or self.target == "litert_rvv"

    @property
    def is_imi(self) -> bool:
        return self.target == "litert_imi"
    
    @property
    def is_aarch64(self) -> bool:
        return self.target == "litert_neoverse" or self.target == "litert_oryon"

    @property
    def xnnpack_root(self) -> Path:
        return (DEV_ENV_ROOT / "XNNPACK")

    @property
    def tflite_x86_build_dir(self) -> Path:
        return self.root / "linux-build-x86"

    def check_deps(self, is_build: bool):
        if not self.xnnpack_root.exists():
            raise RuntimeError(f"LiteRT requires XNNPACK dependency to be initialized. Please initialize with `iminnt -t xnnpack_imi`, then try again.")
        elif is_build and self.target != "litert_x86" and not self.tflite_x86_build_dir.exists():
            raise RuntimeError(f"Cross-compiling LiteRT requires a x86 build first for host-tool usage. Please build with `iminnt -t litert_x86 build`, then try again.")


    @property
    def xnnpack_build_dir(self) -> Path:
        return self.xnnpack_root / self._xnnpack_build_dir

    @property
    def cmake_args(self) -> Dict:
        tf_cxx_flags={
            "TF_MAJOR_VERSION": "2",
            "TF_MINOR_VERSION": "20",
            "TF_PATCH_VERSION": "0",
            "TF_VERSION_SUFFIX": ""
        }
        tf_cxx_flags_str = " ".join([f"-D{k}={v}" for k,v in tf_cxx_flags.items()])
        base_args = {
            # "CMAKE_BUILD_TYPE": "RelWithDebInfo",
            "CMAKE_BUILD_TYPE": "Release",
            "CMAKE_INSTALL_PREFIX": f"{self.install_dir}",
            "TFLITE_ENABLE_XNNPACK": "ON",
            "TFLITE_ENABLE_EXTERNAL_DELEGATE": "OFF",
            "FLATBUFFERS_BUILD_TESTS": "OFF",
        }

        if self.is_riscv or self.is_aarch64:
            arch_args = {
                "CROSS_STATIC": "ON",
                "CMAKE_TOOLCHAIN_FILE": f"{CROSS_TOOLCHAIN_PATH}",
                "HAVE_POSIX_REGEX": "0",
                "HAVE_STEADY_CLOCK": "0",
                "HAVE_STD_REGEX": "0",
                "TFLITE_HOST_TOOLS_DIR": self.tflite_x86_build_dir / "flatbuffers-flatc" / "bin"
            }
        else:
            arch_args = {}
        base_args.update(arch_args)
        base_args["TFLITE_ENABLE_INSTALL"] = "OFF"
        base_args["FETCHCONTENT_SOURCE_DIR_XNNPACK"] = str(self.xnnpack_root)
        if self.is_riscv:
            base_args["LIBM"] = f'{IMI_SDK_ROOT}/sysroot/usr/lib/libm.a'
        if self.is_imi:
            base_args["XNN_ENABLE_RISCV_IMI"] = "0"
        elif self.is_riscv:
            base_args["XNN_ENABLE_RISCV_VECTOR"] = "1"
            base_args["XNN_ENABLE_RISCV_IMI"] = "0"

        if self.is_aarch64:
            base_args["XNN_ENABLE_KLEIDIAI"] = "1"
        base_args["CMAKE_C_FLAGS"] = f'"{tf_cxx_flags_str}"'
        base_args["CMAKE_CXX_FLAGS"] = f'"{tf_cxx_flags_str}"'
        return base_args

    @property
    def install_dir(self) -> Path:
        return self.root / self._install_dir

    @property
    def build_dir(self) -> Path:
        return self.root / self._build_dir

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        cmake_args = " ".join([f"-D{k}={v}" for k,v in self.cmake_args.items()])
        cmds = [ f"rm -rf {self.build_dir}", f"rm -rf {self.install_dir}"]
        return cmds + [
            f"cmake -G Ninja -S {self.tflite_path} -B {self.build_dir} {cmake_args}",
            f"cmake --build {self.build_dir} -j{threads}",
            f"cmake --build {self.build_dir} -j{threads} -t benchmark_model",
        ]

    @property
    def rebuild_cmd(self) -> list:
        return [
            f"cmake --build {self.build_dir} -j",
            f"cmake --build {self.build_dir} -j -t benchmark_model",  
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
        return f"{self.build_dir}/tools/benchmark/benchmark_model"

    @property
    def bench_args(self) -> str:
        bargs = []
        bargs.append("--num_runs=1")
        bargs.append("--num_threads=1")
        bargs.append("--warmup_runs=0")
        bargs.append("--warmup_min_secs=0.0")
        bargs.append("--print_preinvoke_state=true")
        bargs.append("--use_xnnpack=true")
        # test
        bargs.append("--tensor_name_display_length=50")
        bargs.append("--verbose=true")

        return " ".join(bargs)

    @property
    def default_runs(self) -> Dict:
        return {
            "model_test": {"bin": self.bench_bin, "args": f"--graph={self.root}/mobilenet_v1_1.0_224.tflite {self.bench_args}"},
            "mobilenetv4_test": {"bin": self.bench_bin, "args": f"--graph=/projects2/skinzer_models/MobileNetV4-Conv-Large-w8afp32.tflite {self.bench_args}"},
            "3dunet_test": {"bin": self.bench_bin, "args": f"--graph=/projects2/skinzer_models/3dunet_kits19_128x128x128_w8af32.tflite {self.bench_args}"},
            "retinanet_test": {"bin": self.bench_bin, "args": f"--graph=/projects2/skinzer_models/resnext50_32x4d_fpn_w8f32.tflite {self.bench_args}"},
            "model_test_help": {"bin": self.bench_bin, "args": f"--help"},
        }


@register_class("litert_x86")
class LiteRTX86(LiteRTBase):
    def __init__(self):
        super().__init__(
            "litert_x86",
            "linux-build-x86",
            "litert-x86-install",
            "linux-build-x86",
            use_qemu=False
        )

@register_class("litert_imi")
class LiteRTIMI(LiteRTBase):
    def __init__(self):
        super().__init__(
            "litert_imi",
            "linux-build-imi",
            "litert-imi-install",
            "linux-build-imi",
            env=IMI_ENV
        )

@register_class("litert_rvv")
class LiteRTRVV(LiteRTBase):
    def __init__(self):
        super().__init__(
            "litert_rvv",
            "linux-build-rvv",
            "litert-rvv-install",
            "linux-build-rvv",
            env=RVV_ENV
        )

@register_class("litert_oryon")
class LiteRTRVV(LiteRTBase):
    def __init__(self):
        super().__init__(
            "litert_oryon",
            "linux-build-oryon",
            "litert-oryon-install",
            "linux-build-oryon",
            env=ORYON_ENV,
            use_qemu=False,
            remote_info=ORYON_REMOTE
        )