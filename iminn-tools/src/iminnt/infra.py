from typing import Dict, Any, Optional, List
from pathlib import Path
from .utils import shell
from .constants import DEV_ENV_ROOT, BUILD_DEBUG, IMI_RISCV_ARCH, \
    RISCV_ENV_ROOT, RISCV_ENV_INSTALL, IMI_ENV
from .registry import register_class
from .core import BenchSimRunner, CompoundRunner, MultiRunner, SubRunner

BUILD_PILOS_TESTS=False

@register_class("qemu")
class Qemu(BenchSimRunner):

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/csqemu-v9"

    @property
    def target(self) -> str:
        return "qemu"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "csqemu-v9"

    def get_build_cmd(self, threads=None) -> list:
        return [
            "./qemu-build.sh",
            "./qemu-build.sh --system",
        ]

    @property
    def rebuild_cmd(self) -> list:
        return [
            "./qemu-build.sh --rebuild", 
            "./qemu-build.sh --system --rebuild"
        ]

    @property
    def custom_scripts(self) -> List[str]:
        return ["qemu-build.sh"]


@register_class("isaextgen")
class ISAExtGen(BenchSimRunner):

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/isaextgen"

    @property
    def target(self) -> str:
        return "isaextgen"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "isaextgen"

    def get_build_cmd(self, threads=None) -> list:
        return ["./run.sh"]

    @property
    def rebuild_cmd(self) -> list:
        return ["./run.sh"]

@register_class("arctic")
class Arctic(BenchSimRunner):

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/Arctic"

    @property
    def target(self) -> str:
        return "arctic"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "Arctic"

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        return ["rm -rf build", 
                f"cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPERFORMANCE=ON -S {DEV_ENV_ROOT / 'Arctic'}", 
                f"cmake --build build -j{threads}"]

    @property
    def rebuild_cmd(self) -> list:
        return [f"cmake --build {DEV_ENV_ROOT/'Arctic'/'build'} -j"]


@register_class("pilos")
class Pilos(BenchSimRunner):

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/Pilos"

    @property
    def target(self) -> str:
        return "pilos"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "Pilos"

    @property
    def pull_cmd(self) -> Optional[List[str]]:
        return ["./pilos_update.sh"]
    
    @property
    def env(self) -> Dict[str, str]: # Clang compilation fails for some reason
        return {
            "CC": "gcc",
            "CXX": "g++"
        }
    
    def get_test_files(self):
        files = []
        base_path = (self.root / "Tests" / "compile" / "matmul")
        for f in base_path.rglob("*"):
            if f.is_file():
                files.append(f.relative_to(base_path))
        return files
    
    @property
    def clang_bin(self) -> str:
        llvm_path = IMI_ENV["IMI_LLVM_PATH"]
        return f"{llvm_path}/bin/clang++"

    @property
    def cross_arch(self):
        return IMI_ENV["CROSS_ARCH"]

    @property
    def cross_cpu_variant(self):
        return IMI_ENV["CROSS_CPU"]

    @property
    def cross_rvv_len(self):
        return IMI_ENV("RVV_VLEN")

    @property
    def cross_triple(self):
        return IMI_ENV["CROSS_TRIPLE"]

    @property
    def cross_sysroot(self):
        return IMI_ENV["CROSS_SYSROOT"]

    @property
    def cross_toolchain(self):
        return IMI_ENV["CROSS_TOOLCHAIN"]

    def get_sysroot_args(self):
        return f"--sysroot={self.cross_sysroot}"

    def build_tests_cmds(self):
        files = self.get_test_files()
        paths = []
        cmds = []
        rvv_arch_flags = f"-march={self.cross_cpu_variant} -mabi=lp64d"

        for f in files:
            build_path = DEV_ENV_ROOT / 'Pilos' / 'build' / f.parent

            if str(build_path) not in paths:
                cmds.append(f"mkdir -p {build_path}")
                paths.append(build_path)

            cc_cmd = f"{self.clang_bin} -g -static -O2 -Wall"
            cc_cmd = f"{cc_cmd} {self.get_sysroot_args()}"
            cc_cmd = f"{cc_cmd} --target={self.cross_triple}"
            cc_cmd = f"{cc_cmd} {rvv_arch_flags}"
            cc_cmd = f"{cc_cmd} -I{self.root}/Tests" 
            cc_cmd = f"{cc_cmd} {self.root}/Tests/compile/matmul/{f} -o {build_path}/{f.stem}"
            cmds.append(cc_cmd)
        return cmds

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        
        cmds = ["rm -rf build",
                f"./pilos_opgen.sh",
                f"cmake -G Ninja DCMAKE_CXX_FLAGS=\"-std=c++17 -Wall\" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPERFORMANCE=ON -B build -S {DEV_ENV_ROOT/'Pilos'}",
                f"cmake --build {DEV_ENV_ROOT/'Pilos'/'build'} -j{threads}",
                ]
        if BUILD_PILOS_TESTS:
            cmds = cmds + self.build_tests_cmds()
        return cmds

    @property
    def rebuild_cmd(self) -> list:
        return [f"cmake --build {DEV_ENV_ROOT/'Pilos'/'build'} -j"]

    @property
    def custom_scripts(self) -> List[str]:
        return ["pilos_update.sh", "pilos_opgen.sh"]

@register_class("permafrost")
class Permafrost(BenchSimRunner):

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/Permafrost"

    @property
    def target(self) -> str:
        return "permafrost"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "Permafrost"

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        return ["rm -rf build", 
                f"cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DPERFORMANCE=ON -S {DEV_ENV_ROOT / 'Permafrost'}", 
                f"cmake --build {DEV_ENV_ROOT/'Permafrost'/'build'} -j{threads}"]

    @property
    def rebuild_cmd(self) -> list:
        return [f"cmake --build {DEV_ENV_ROOT/'Permafrost'/'build'} -j"]

@register_class("spike")
class Spike(BenchSimRunner):

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/IMachines_Spike"

    @property
    def target(self) -> str:
        return "spike"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "IMachines_Spike"
    
    @property
    def env(self) -> Dict[str, str]:
        return {
            "CXXFLAGS": f"-I{DEV_ENV_ROOT / 'Arctic' / 'include'} -I{DEV_ENV_ROOT / 'Permafrost' / 'include'}",
            "CFLAGS": f"-I{DEV_ENV_ROOT / 'Arctic' / 'include'} -I{DEV_ENV_ROOT / 'Permafrost' / 'include'}",
        }

    def get_build_cmd(self, threads=None) -> list:
        return ["rm -rf build", "./spike_build.sh"]

    @property
    def rebuild_cmd(self) -> list:
        return [f"make -C build"]

    @property
    def custom_scripts(self) -> List[str]:
        return ["spike_build.sh"]

@register_class("riscv_toolchain")
class RISCVToolchain(SubRunner):

    def __init__(self, root: Path = RISCV_ENV_ROOT, install_path: Path = RISCV_ENV_INSTALL):
        self.install_path = install_path
        super().__init__(root)

    @property
    def remote_path(self):
        return "https://github.com/riscv/riscv-gnu-toolchain"

    @property
    def target(self) -> str:
        return "riscv_toolchain"
    
    @property
    def sub_root(self) -> str:
        return "riscv-gnu-toolchain"

    @property
    def pull_cmd(self) -> Optional[List[str]]:
        return ["echo 'Already up to date'"]

    @property
    def git_tag(self) -> str:
        return "2024.11.22"

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        return [
            f"rm -f config.status",
            f"rm -f config.log",
            f"./configure --prefix={self.install_path} --with-arch={IMI_RISCV_ARCH.lower().replace('_ximimce', '')} --with-abi=lp64d",
            f"make clean",
            f"make -j{threads} linux"
        ]

    @property
    def rebuild_cmd(self) -> list:
        return ["make -j linux"]


@register_class("llvm")
class LLVM(SubRunner):
    
    def __init__(self, root: Path = RISCV_ENV_ROOT, install_path: Path = RISCV_ENV_INSTALL):
        self.install_path = install_path
        super().__init__(root)

    # TODO: This should probably be a full path
    @property
    def build_dir(self):
        return "llvm-build"

    @property
    def target(self) -> str:
        return "llvm"

    @property
    def remote_path(self):
        return "https://github.com/I-Machines/llvm-project"
    
    @property
    def sub_root(self) -> str:
        return "llvm-project"

    def has_ccache(self):
        return shell(["which", "ccache"], no_fail=True, silent=True)["exit_code"] == 0

    @property
    def cmake_args(self):
        return {
            "CMAKE_BUILD_TYPE": "Release",
            "LLVM_CCACHE_BUILD": "ON" if self.has_ccache() else "OFF",
            "LLVM_FORCE_ENABLE_STATS": "ON",
            "LLVM_ENABLE_PROJECTS": "clang;lld;mlir;openmp",
            "LLVM_OPTIMIZED_TABLEGEN": "ON",
            "LLVM_ENABLE_ASSERTIONS": "ON",
            "LLVM_FORCE_ENABLE_STATS": "ON",
            "BUILD_SHARED_LIBS": "True",
            "LLVM_PARALLEL_LINK_JOBS": "1",
            "LLVM_TARGETS_TO_BUILD": "RISCV",
            "CMAKE_INSTALL_PREFIX": f"{self.install_path}",
            "LLVM_USE_SPLIT_DWARF": "True",
            "LLVM_BUILD_TESTS": "False",
            "DEFAULT_SYSROOT": f"{self.install_path}/sysroot",
            "LLVM_DEFAULT_TARGET_TRIPLE": "riscv64-unknown-linux-gnu"
        }

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""

        cmake_args = " ".join([f"-D{k}={v}" for k,v in self.cmake_args.items()])
        return [
            f"rm -rf {self.build_dir}",
            f"cmake -G Ninja -S {self.root}/llvm -B {self.root}/{self.build_dir} {cmake_args} {self.root}/llvm",
            f"cmake --build {self.root}/{self.build_dir} -j{threads} -t install"
        ]

    @property
    def rebuild_cmd(self) -> list:
        return [f"cmake --build {self.root}/{self.build_dir} -j -t install"]

@register_class("riscv-env")
class RISCVEnv(CompoundRunner):
    def __init__(self):
        root_dir = RISCV_ENV_ROOT # Contains all sub directories
        self.install_path = RISCV_ENV_INSTALL # Contains all sub directories
        sub_runners = [
                        RISCVToolchain(root_dir, self.install_path), 
                        LLVM(root_dir, self.install_path)]
        super().__init__(root_dir, sub_runners)

    @property
    def apt_reqs(self) -> List[str]:
        return [
            "ninja-build",
            "autoconf",
            "automake",
            "autotools-dev",
            "curl",
            "libmpc-dev",
            "libmpfr-dev",
            "libgmp-dev",
            "gawk",
            "build-essential",
            "bison",
            "flex",
            "texinfo",
            "gperf",
            "libtool" ,
            "patchutils",
            "bc",
            "zlib1g-dev",
            "libexpat-dev"
        ]

    @property
    def target(self) -> str:
        return "riscv-env"

    def get_build_cmd(self, threads=None) -> list:
        return [f"rm -rf {self.install_path}", 
                f"mkdir -p {self.root}/riscv"]

    @property
    def rebuild_cmd(self) -> list:
        return []


@register_class("simpoint")
class Simpoint(BenchSimRunner):

    @property
    def remote_path(self):
        return "/projects/performance/tools/simpoint"
    
    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "simpoint"

    @property
    def target(self) -> str:
        return "simpoint"

    def get_build_cmd(self, threads=None) -> list:
        return []

    @property
    def rebuild_cmd(self) -> list:
        return []

@register_class("psim")
class PSim(MultiRunner):
    def __init__(self):
        # IMPORTANT: The order of these matters, roughly reflecting the dependency chain.
        super().__init__("psim", [RISCVEnv(), Arctic(), Permafrost(), Pilos(), Spike(), Qemu(), Simpoint()])
