
from typing import Dict, Any, Optional, List
from pathlib import Path
from .constants import DEV_ENV_ROOT, NPU_ENV_ROOT, NPU_ENV_INSTALL, IMI_SDK_ROOT, IMI_RISCV_ARCH, CROSS_TOOLCHAIN_PATH
from .registry import register_class
from .core import CompoundRunner, SubRunner

@register_class("linux-kernel")
class LinuxKernel(SubRunner):
    
    def __init__(self, root: Path = NPU_ENV_ROOT, install_path: Path = NPU_ENV_INSTALL):
        self.install_path = install_path
        super().__init__(root)

    @property
    def build_dir(self):
        return "build"

    @property
    def target(self) -> str:
        return "linux-kernel"

    @property
    def remote_path(self):
        return "/projects/datasets/VSI/npu-drv-7.0.0.66/linux-6.6.37"
    
    @property
    def sub_root(self) -> str:
        return "linux"

    @property
    def cmake_args(self):
        return {}

    def get_build_cmd(self, threads=None) -> list:
        return []

    @property
    def rebuild_cmd(self) -> list:
        return []
    
    @property
    def post_init(self) -> List[str]:
        img_name = "Image-6.6.37"
        img_path = Path(self.remote_path).parent / img_name
        return [
            f"cp {img_path} {self.root}/{img_name}"
        ]


@register_class("npu-driver")
class NPUDriver(SubRunner):
    
    def __init__(self, root: Path = NPU_ENV_ROOT, install_path: Path = NPU_ENV_INSTALL):
        self.install_path = install_path
        super().__init__(root)

    @property
    def build_dir(self):
        return "npu-build"

    @property
    def target(self) -> str:
        return "npu-driver"

    @property
    def remote_path(self):
        return "/projects/datasets/VSI/npu-drv-7.0.0.66/npu-drv-src-7.0.0.66.tgz"
    
    @property
    def sub_root(self) -> str:
        return "npu_driver"

    @property
    def custom_scripts(self) -> List[str]:
        return ["npu-build.sh"]

    @property
    def cmake_args(self):
        return {}

    def get_build_cmd(self, threads=None) -> list:
        return [f"./npu-build.sh --clean --build-dir {self.build_dir} --toolchain {IMI_SDK_ROOT} --linux-kernel {NPU_ENV_ROOT}/linux"]

    @property
    def rebuild_cmd(self) -> list:
        return [f"./npu-build.sh --build-dir {self.build_dir} --toolchain {IMI_SDK_ROOT} --linux-kernel {NPU_ENV_ROOT}/linux"]


@register_class("npu-env")
class NPUEnv(CompoundRunner):
    def __init__(self):
        root_dir = NPU_ENV_ROOT # Contains all sub directories
        self.install_path = NPU_ENV_INSTALL # Contains all sub directories
        sub_runners = [
                        LinuxKernel(root_dir, self.install_path),
                        NPUDriver(root_dir, self.install_path)
                    ]
        super().__init__(root_dir, sub_runners)

    @property
    def target(self) -> str:
        return "npu-env"

    def get_build_cmd(self, threads=None) -> list:
        return [f"mkdir -p {self.root}/npu"]

    @property
    def rebuild_cmd(self) -> list:
        return []