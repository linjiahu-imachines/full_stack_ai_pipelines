from typing import Dict, Optional, List, Tuple
from pathlib import Path
from .constants import IMI_SDK_ROOT, DEV_ENV_ROOT, IMI_RISCV_ARCH, CROSS_TOOLCHAIN_PATH, IMI_ENV, NATIVE_ENV, USE_DEBUG_SYMS
from .registry import register_class
from .core import BenchSimRunner

OPT_LEVEL = "O2"

class PlaygroundBase(BenchSimRunner):
    def __init__(self, 
                suffix: str,
                env: Dict,
                remote_info: Optional[Dict] = None,
                use_qemu: bool = True):
        assert "CROSS_TRIPLE" in env
        assert "CROSS_SYSROOT" in env
        assert "IMI_LLVM_PATH" in env
        self._env = env
        self._target = f"playground-{suffix}"
        self._suffix = suffix
        self._remote_info = remote_info
        self._use_qemu = use_qemu
        self._dir = f"{self._target}-build"

    @property
    def remote_path(self):
        return str(DEV_ENV_ROOT / "playground")

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
    def is_aarch64(self) -> bool:
        return False
    
    @property
    def install_dir(self) -> Path:
        return self.root / self._dir

    @property
    def build_dir(self) -> Path:
        return self.root / self._dir

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "playground"

    @property
    def test_bin(self) -> str:
        return f"{self.install_dir}/playground"
    
    @property
    def default_bin(self) -> str:
        return self.test_bin

    @property
    def use_qemu(self) -> bool:
        return self._use_qemu

    @property
    def remote_info(self) -> Optional[Dict]:
        return self._remote_info
    
    def get_executables(self):
        execs = []
        for f in self.root.iterdir():
            # Ignore non-files, libraries, and files starting with "_"
            if f.is_file() and f.suffix == ".c" and not f.stem.startswith("utils") and not f.stem.startswith("_"):
                execs.append(f.stem)
        return execs

    def get_libs(self):
        libs = []
        for f in self.root.iterdir():
            if f.is_file() and f.suffix == ".c" and f.stem.startswith("utils"):
                libs.append(f.stem)
        return libs

    @property
    def default_runs(self) -> Dict:
        runs = {}
        for p in self.get_executables():
            runs[p] = {"bin": f"{self.install_dir}/{p}", "args": ""}
        return runs

    @property
    def clang_bin(self) -> str:
        llvm_path = self.env["IMI_LLVM_PATH"]
        return f"{llvm_path}/bin/clang"

    @property
    def arch(self):
        return self.env["CROSS_ARCH"]

    @property
    def cpu_variant(self):
        return self.env["CROSS_CPU"]

    @property
    def rvv_len(self):
        return self.env.get("RVV_VLEN")

    @property
    def triple(self):
        return self.env["CROSS_TRIPLE"]

    @property
    def sysroot(self):
        return self.env["CROSS_SYSROOT"]

    @property
    def toolchain(self):
        return self.env["CROSS_TOOLCHAIN"]
    
    def get_sysroot_args(self):
        return f"--sysroot={self.sysroot}"

    def get_arch_flags(self):
        if self.arch == "aarch64":
            return f"-mcpu={self.cpu_variant}"
        elif self.arch == "riscv64":
            f = f"-march={self.cpu_variant}"
            if self.rvv_len is not None:
                f = f"{f} -mrvv-vector-bits={self.rvv_len}"
            return f"{f} -mabi=lp64d"
        else:
            assert self.arch == "x86_64"
            return f"-march=x86-64 -mtune={self.cpu_variant}"
    
    def get_cc_cmds(self):
        c_execs = self.get_executables()
        c_libs = self.get_libs()
        cc_cmds = []
        
        # First, compile utilities
        link_libs = []
        for p in c_libs:
            cc_cmd = f"{self.clang_bin} -g -{OPT_LEVEL} -Wall"
            cc_cmd = f"{cc_cmd} {self.get_sysroot_args()}"
            cc_cmd = f"{cc_cmd} --target={self.triple}"
            cc_cmd = f"{cc_cmd} {self.get_arch_flags()}"
            cc_cmd = f"{cc_cmd} -I{self.root}"
            ll = f"{self.build_dir}/{p}.o"
            cc_cmd = f"{cc_cmd} -c {self.root}/{p}.c -o {ll} -lm"
            link_libs.append(ll)
            cc_cmds.append(cc_cmd)

        ll_str = " ".join(link_libs)
        for p in c_execs:
            cc_cmd = f"{self.clang_bin} -g -static -{OPT_LEVEL} -Wall"
            cc_cmd = f"{cc_cmd} {self.get_sysroot_args()}"
            cc_cmd = f"{cc_cmd} --target={self.triple}"
            cc_cmd = f"{cc_cmd} {self.get_arch_flags()}"
            cc_cmd = f"{cc_cmd} -I{self.root}" 
            cc_cmd = f"{cc_cmd} {self.root}/{p}.c {ll_str} -o {self.build_dir}/{p} -lm"
            cc_cmds.append(cc_cmd)
        return cc_cmds
        
    def get_build_cmd(self, threads=None) -> list:
        # First, clean
        cmds = [
            f"rm -rf {self.build_dir}",
            f"mkdir -p {self.build_dir}",
        ] + self.get_cc_cmds()
        return cmds

    @property
    def rebuild_cmd(self) -> list:
        return self.get_cc_cmds()

@register_class("playground-x86")
class PlaygroundX86(PlaygroundBase):
    def __init__(self):
        super().__init__(
            "x86",
            NATIVE_ENV,
            use_qemu=False
        )


@register_class("playground-riscv")
class PlaygroundRISCV(PlaygroundBase):
    def __init__(self):
        super().__init__(
            "riscv",
            IMI_ENV
        )

    @property
    def is_riscv(self) -> bool:
        return True