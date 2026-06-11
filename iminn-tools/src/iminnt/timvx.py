from typing import Dict, Any, Optional, List
from pathlib import Path
from .constants import DEV_ENV_ROOT, NPU_ENV_ROOT, NPU_ENV_INSTALL, IMI_SDK_ROOT, \
    IMI_RISCV_ARCH, CROSS_TOOLCHAIN_PATH, NPU_SDK, IMI_ENV
from .registry import register_class
from .core import BenchSimRunner

### Reference for build CMAKE args:
# | option name                 | Summary                                                                                  | Default |
# | -----                       | -----                                                                                    | ----- |
# |TIM_VX_ENABLE_TEST           | Enable unit test case for public APIs and ops                                            | OFF |
# |TIM_VX_ENABLE_LAYOUT_INFER   | Build with tensor data layout inference support                                          | ON |
# |TIM_VX_USE_EXTERNAL_OVXLIB   | Replace internal with a prebuilt libovxlib library                                       | OFF |
# |OVXLIB_LIB                   | full path to libovxlib.so include so name, required if `TIM_VX_USE_EXTERNAL_OVXLIB`=ON   | Not set |
# |OVXLIB_INC                   | ovxlib's include path, required if `TIM_VX_USE_EXTERNAL_OVXLIB`=ON                       | Not set |
# |EXTERNAL_VIV_SDK             | Give external vivante openvx driver libraries                                            | Not set|
# |TIM_VX_BUILD_EXAMPLES        | Build example applications                                                               | OFF |
# |TIM_VX_ENABLE_40BIT          | Enable large memory (over 4G) support in NPU driver                                      | OFF |
# |TIM_VX_ENABLE_PLATFORM       | Enable multi devices support                                                             | OFF |
# |TIM_VX_ENABLE_PLATFORM_LITE  | Enable lite multi-device support, only work when `TIM_VX_ENABLE_PLATFORM`=ON             | OFF |
# |VIP_LITE_SDK                 | full path to VIPLite sdk, required when `TIM_VX_ENABLE_PLATFORM_LITE`=ON                 | Not set |
# |TIM_VX_ENABLE_GRPC           | Enable gPRC support, only work when `TIM_VX_ENABLE_PLATFORM`=ON                          | OFF |
# |TIM_VX_DBG_ENABLE_TENSOR_HNDL| Enable built-in tensor from handle                                                       | ON |
# |TIM_VX_ENABLE_TENSOR_CACHE   | Enable tensor cache for const tensor, check [OpenSSL build notes](docs/openssl_build.md) | OFF |

class TIMVXBase(BenchSimRunner):

    def __init__(self, 
                target: str,
                build_dir: str, 
                install_dir: str, 
                env: Optional[Dict] = None, 
                remote_info: Optional[Dict] = None, 
                use_qemu: bool = True,
                build_ref: bool = True):
        self._build_dir = build_dir
        self._install_dir = install_dir
        self.build_ref = build_ref
        self._env = env or {}
        self._target = target
        self._remote_info = remote_info
        self._use_qemu = use_qemu

    @property
    def custom_scripts(self) -> List[str]:
        return ["type_header_fix0.patch"]

    @property
    def is_riscv(self) -> bool:
        return False
    
    @property
    def is_aarch64(self) -> bool:
        return False
    
    @property
    def remote_path(self):
        return "https://github.com/VeriSilicon/TIM-VX"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "timvx"

    # TODO: add models to download for testing
    @property
    def post_init(self) -> List[str]:
        return ["git apply type_header_fix0.patch"]

    @property
    def install_dir(self) -> Path:
        return self.root / self._install_dir

    @property
    def build_dir(self) -> Path:
        return self.root / self._build_dir

    @property
    def target(self) -> str:
        return "timvx"

    @property
    def use_qemu(self) -> bool:
        return self._use_qemu
    
    @property
    def remote_info(self) -> Optional[Dict]:
        return self._remote_info

    @property
    def env(self) -> Dict:
        return self._env

    @property
    def cmake_args(self):
        if self.is_riscv:
            base_args = {
                # "CROSS_STATIC": "ON",
                "CMAKE_TOOLCHAIN_FILE": f"{CROSS_TOOLCHAIN_PATH}",
                "EXTERNAL_VIV_SDK": f"{NPU_SDK}",
                # "BUILD_SHARED_LIBS": "OFF",
                "TIM_VX_USE_EXTERNAL_OVXLIB": "ON",
                "OVXLIB_LIB": f"{NPU_SDK}/drivers/libovxlib.so",
                "OVXLIB_INC": f"{NPU_ENV_ROOT}/npu_driver/src/Vivante_ML_Toolkit_OVXLIB_dev/include",   
            }
        else:
            # Only options are x86 and riscv
            base_args = {}
        
        base_args['CMAKE_BUILD_TYPE'] = "Release"
        base_args['CMAKE_INSTALL_PREFIX'] = f"{self.install_dir}"
        base_args["TIM_VX_BUILD_EXAMPLES"] = "ON"

        return base_args

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        cmake_args = " ".join([f"-D{k}={v}" for k,v in self.cmake_args.items()])
        return [
            f"rm -rf {self.build_dir}",
            f"rm -rf {self.install_dir}",
            f"cmake -G Ninja -S {self.root} -B {self.build_dir} {cmake_args}",
            f"cmake --build {self.build_dir} -j{threads} -t install"
        ]

    @property
    def rebuild_cmd(self) -> list:
        return [f"cmake --build {self.build_dir} -j -t install"]


@register_class("timvx_x86")
class TIMVXX86(TIMVXBase):
    def __init__(self):
        super().__init__(
            "timvx_x86",
            "timvx-build-x86",
            "timvx-x86-install",
            use_qemu=False
        )

@register_class("timvx_riscv")
class TIMVXRiscv(TIMVXBase):

    def __init__(self):
        super().__init__(
            "timvx_riscv",
            "timvx-build-riscv",
            "timvx-riscv-install",
            env=IMI_ENV
        )

    @property
    def is_riscv(self) -> bool:
        return True

    @classmethod
    def clang_root(cls) -> Path:
        return IMI_SDK_ROOT