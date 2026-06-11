from pathlib import Path
import re
from typing import List, Dict, Optional
import json

from .constants import IMI_SDK_ROOT, CROSS_TOOLCHAIN_PATH, DEV_ENV_ROOT, IMI_RISCV_ARCH, AZURE_KEY_PATH, \
    AZURE_BENCH_PATH, XNNPACK_RESOURCES, IMI_ENV, RVV_ENV, NEOVERSE_REMOTE, NEOVERSE_ENV, AMX_REMOTE, AMX_ENV, USE_DEBUG_SYMS
from .registry import register_class
from .core import BenchSimRunner

BUILD_RM_OLD_KERNELS=False

### XNNPACK


class XNNPACKBase(BenchSimRunner):
    BASE_BENCH_FILTER = "--benchmark_min_time=1x --benchmark_min_warmup_time=0 --benchmark_dry_run=false --benchmark_repetitions=1 --num_threads=1"
    BENCH_NAMES=["shufflenet_v1_g8/M:49/N:96/K:48/real_time", "shufflenet_v1_g4/M:49/N:68/K:272/real_time", "shufflenet_v1_g8/M:49/N:48/K:192/real_time"]

    def __init__(self, 
                target: str,
                build_dir: Path, 
                install_dir: Path, 
                ukernels: List[str], 
                env: Optional[Dict] = None, 
                remote_info: Optional[Dict] = None, 
                use_qemu: bool = True):
        self._build_dir = build_dir
        self._install_dir = install_dir
        self._ukernels = ukernels
        self._env = env or {}
        self._target = target
        self._remote_info = remote_info
        self._use_qemu = use_qemu
    
    @property
    def remote_path(self):
        return "https://github.com/I-Machines/XNNPACK"
    
    @property
    def upstream_remote(self) -> Optional[str]:
        return "https://github.com/google/XNNPACK"

    @property
    def post_init(self) -> List[str]:
        return [
            f"git -C {self.root} remote add upstream https://github.com/google/XNNPACK",
            f"git -C {self.root} checkout upstream",
            f"git -C {self.root} checkout dev-staging",
        ]

    @property
    def ukernels(self) -> List[str]:
        return self._ukernels
    
    @property
    def install_dir(self) -> Path:
        return self.root / self._install_dir

    @property
    def build_dir(self) -> Path:
        return self.root / self._build_dir

    @property
    def env(self) -> Dict:
        return self._env

    @property
    def target(self) -> str:
        return self._target

    @property
    def is_riscv(self) -> bool:
        return self.target == "xnnpack_imi" or self.target == "xnnpack_rvv"
    
    @property
    def is_imi(self) -> bool:
        return self.target == "xnnpack_imi"

    @property
    def is_aarch64(self) -> bool:
        return self.target == "xnnpack_neoverse"

    @property
    def root(self) -> str:
        return DEV_ENV_ROOT / "XNNPACK"

    def get_build_cmd(self, threads=None) -> list:
        threads = threads if threads is not None else ""
        cmake_args = " ".join([f"-D{k}={v}" for k,v in self.cmake_args.items()])
        cmds =  [f"rm -rf {self.build_dir}",
                 f"rm -rf {self.install_dir}",
                 "./scripts/generate-qs8-gemm.sh",
                 "./scripts/generate-qs8-igemm.sh",
                 "./scripts/generate-tests.sh",
                 f"./tools/update-microkernels.py --output {self.root}",
                 f"cmake -G Ninja -S {self.root} -B {self.build_dir} {cmake_args}",
                 f"cmake --build {self.build_dir} -j{threads} -t install"
                ]
        if self.is_imi and BUILD_RM_OLD_KERNELS:
            rm_kernel_cmds = [] 
            with open(f"{self.root}/cmake/gen/imi_microkernels.cmake", "r") as f:
                for l in f.readlines():
                    if l.strip().startswith("src"):
                        kp = l.strip().replace(")", "")
                        if Path(f"{self.root}/{kp}").exists():
                            rm_kernel_cmds.append(f"rm {self.root}/{kp}")
            cmds = rm_kernel_cmds + cmds
        return cmds

    @property
    def rebuild_cmd(self) -> list:
        return [f"cmake --build {self.build_dir} -j -t install"]

    @property
    def test_bin(self) -> str:
        return f"{self.build_dir}/test/qd8-f32-qc8w-gemm-minmax-test"

    @property
    def igemm_test_bin(self) -> str:
        return f"{self.build_dir}/test/qd8-f32-qc8w-igemm-minmax-test"

    @property
    def igemm_rq_test_bin(self) -> str:
        return f"{self.build_dir}/test/qs8-qc8w-igemm-minmax-fp32-test"
    
    @property
    def gemm_test_bin(self) -> str:
        return f"{self.build_dir}/test/qd8-f32-qc8w-gemm-minmax-test"

    @property
    def gemm_rq_test_bin(self) -> str:
        return f"{self.build_dir}/test/qs8-qc8w-gemm-minmax-fp32-test"

    @property
    def igemm_qc4w_test_bin(self) -> str:
        return f"{self.build_dir}/test/qd8-f32-qc4w-igemm-minmax-test"

    @property
    def igemm_rq_qc4w_test_bin(self) -> str:
        return f"{self.build_dir}/test/qs8-qc4w-igemm-minmax-fp32-test"
    
    @property
    def gemm_qc4w_test_bin(self) -> str:
        return f"{self.build_dir}/test/qd8-f32-qc4w-gemm-minmax-test"

    @property
    def gemm_rq_qc4w_test_bin(self) -> str:
        return f"{self.build_dir}/test/qs8-qc4w-gemm-minmax-fp32-test"

    @property
    def bmm_test_bin(self) -> str:
        return f"{self.build_dir}/test/subgraph/batch-matrix-multiply-test"

    @property
    def bench_bin(self) -> str:
        return f"{self.build_dir}/bench/qd8-f32-qc8w-gemm-bench"

    @property
    def sweep_bin(self) -> str:
        return self.bench_bin

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
    def cmake_args(self):
        base_args = {
            "CMAKE_BUILD_TYPE": "RelWithDebInfo",
            "HAVE_POSIX_REGEX": "0",
            "HAVE_STEADY_CLOCK": "0",
            "HAVE_STD_REGEX": "0",
            "XNNPACK_BUILD_TESTS": "ON",
            "XNNPACK_BUILD_LIBRARY": "ON",
            "CMAKE_INSTALL_PREFIX": f"{self.install_dir}",
            "XNNPACK_BUILD_BENCHMARKS": "ON"
        }
        if self.is_riscv or self.is_aarch64:
            base_args.update({
                "CROSS_STATIC": "ON",
                "CMAKE_TOOLCHAIN_FILE": f"{CROSS_TOOLCHAIN_PATH}",
            })
        else:
            base_args.update({
                "XNN_ENABLE_RISCV_IMI": "OFF",
            })

        if self.is_riscv:
            base_args["LIBM"] = f'{IMI_SDK_ROOT}/sysroot/usr/lib/libm.a'
        if self.is_imi:
            base_args["XNN_ENABLE_RISCV_IMI"] = "1"
        elif self.is_riscv:
            base_args["XNN_ENABLE_RISCV_VECTOR"] = "1"
            base_args["XNN_ENABLE_RISCV_IMI"] = "0"

        if self.is_aarch64:
            base_args["XNN_ENABLE_KLEIDIAI"] = "1"

        if USE_DEBUG_SYMS:
            base_args["DEBUG_SYMS"] = "ON"
            
        return base_args
    
    def add_sim_args(self, sim_dir: Path, bin_path, bin_args):
        if "benchmark" in bin_args and "benchmark_out" not in bin_args:
            out_fmt = "--benchmark_out_format=json" if "benchmark_out_format" not in bin_args else ""
            # Set output directory to local, that way we can reuse the conditional below
            remote_bin_args = f"{bin_args} {out_fmt} --benchmark_out={sim_dir}/stats.json"
        
        if "benchmark_out" in bin_args:
            bench_path = None
            for sba in bin_args.split():
                if "--benchmark_out=" in sba:
                    bench_path = sba.replace("--benchmark_out=", "").strip()
                    break
            assert bench_path is not None, f"Unable to find output path for {remote_bin_args}"
            local_results_fpath = bench_path
            results_fpath = bench_dir / "stats.json"
            bin_args = remote_bin_args.replace(local_results_fpath, str(remote_results_fpath))

    def get_sim_results(self, sim_dir: Path, return_perf_info = False):
        stat_file = (sim_dir / "stats.json")
        if not stat_file.exists():
            return None
        with open(str(stat_file), "r") as f:
            data = json.load(f)
        # Just grab the first result
        bench = data["benchmarks"]
        assert len(bench) > 0, f"Expected at least one benchmark result in {stat_file}"
        outputs = {}
        for b in bench:
            name = b["name"].split("/")[0]
            time_ns = float(b["cpu_time"])
            if "cpufreq" not in b:
                assert "context" in data and "mhz_per_cpu" in data["context"]
                cpu_freq = float(data["context"]["mhz_per_cpu"] / (10**3))
            else:
                cpu_freq = float(b["cpufreq"]) / (10**9)
            cycles = int(time_ns * cpu_freq)
            if return_perf_info:
                outputs[name] = {"cycles": cycles, "time_ns": time_ns, "freq": cpu_freq}
            else:
                outputs[name] = time_ns
        return outputs

    def get_kernels(self):
        assert isinstance(self.ukernels, list)
        ukernels = []
        for uk in self.ukernels:
            kpath = XNNPACK_RESOURCES / uk
            assert kpath.exists(), f"Unable to find kernel file at {kpath}"
            with open(str(kpath), "r") as f:
                kernels = f.readlines()
            ukernels += [k.strip() for k in kernels]
        return ukernels

    @property
    def default_runs(self) -> Dict:
        base_defaults = {}
        ukernels = self.get_kernels()
        benches = 0
        for uk_ in ukernels:
            uk = uk_.strip()
            s = re.search(".*_gemm_minmax_ukernel_(\d+)x(\d+)(c(\d+))?__(.*)", uk)
            if s is None:
                raise ValueError(f"Unable to parse kernel name {uk}")
            mt = s.group(1)
            nt = s.group(2)
            kt = s.group(4) or 1
            variant = s.group(5).replace("asm_", "")
            base_key = f"gemm_{mt}x{nt}c{kt}_{variant}"
            igemm_base_key = f"igemm_{mt}x{nt}c{kt}_{variant}"
            # First, add benchmarks for each kernel
            for i, b in enumerate(XNNPACKBase.BENCH_NAMES):
                # For simplicity, add a numbered variant
                base_defaults[f"{base_key}_bench{i}"] = {"bin": self.bench_bin, "args": f"{XNNPACKBase.BASE_BENCH_FILTER} --benchmark_filter={uk}/{b}"}
                base_defaults[f"gemm_bench{benches}"] = {"bin": self.bench_bin, "args": f"{XNNPACKBase.BASE_BENCH_FILTER} --benchmark_filter={uk}/{b}"}
                benches += 1
            gemm_test_filter = uk.replace("_ukernel", "").upper()
            gemm_rq_test_filter = uk.replace("qd8_f32_qc8w_gemm_minmax", "qs8_qc8w_gemm_minmax_fp32").replace("_ukernel", "").upper()
            igemm_test_filter = uk.replace("_ukernel", "").replace("gemm", "igemm").upper()
            igemm_rq_test_filter = uk.replace("qd8_f32_qc8w_gemm_minmax", "qs8_qc8w_gemm_minmax_fp32").replace("_ukernel", "").replace("gemm", "igemm").upper()

            gemm_qc4w_test_filter = gemm_test_filter.replace("QC8W", "QC4W").replace("C16", "C8").upper()
            gemm_qc4w_rq_test_filter = gemm_rq_test_filter.replace("QC8W", "QC4W").replace("C16", "C8").upper()
            igemm_qc4w_test_filter = igemm_test_filter.replace("QC8W", "QC4W").replace("C16", "C8").upper()
            igemm_qc4w_rq_test_filter = igemm_rq_test_filter.replace("QC8W", "QC4W").replace("C16", "C8").upper()

            base_defaults[f"{base_key}_test"] = {"bin": self.gemm_test_bin, "args": f"--gtest_filter=*{gemm_test_filter}/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
            base_defaults[f"{igemm_base_key}_test"] = {"bin": self.igemm_test_bin, "args": f"--gtest_filter=*{igemm_test_filter}/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
            base_defaults[f"{base_key}_rq_test"] = {"bin": self.gemm_rq_test_bin, "args": f"--gtest_filter=*{gemm_rq_test_filter}/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
            base_defaults[f"{igemm_base_key}_rq_test"] = {"bin": self.igemm_rq_test_bin, "args": f"--gtest_filter=*{igemm_rq_test_filter}/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}

            # Q4
            base_q4_key = base_key.replace("c16", "c8")
            igemm_q4_base_key = igemm_base_key.replace("c16", "c8")
            base_defaults[f"{base_q4_key}_qc4w_test"] = {"bin": self.gemm_qc4w_test_bin, "args": f"--gtest_filter=*{gemm_qc4w_test_filter}/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
            base_defaults[f"{igemm_q4_base_key}_qc4w_test"] = {"bin": self.igemm_qc4w_test_bin, "args": f"--gtest_filter=*{igemm_qc4w_test_filter}/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
            base_defaults[f"{base_q4_key}_rq_qc4w_test"] = {"bin": self.gemm_rq_qc4w_test_bin, "args": f"--gtest_filter=*{gemm_qc4w_rq_test_filter}/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
            base_defaults[f"{igemm_q4_base_key}_rq_qc4w_test"] = {"bin": self.igemm_rq_qc4w_test_bin, "args": f"--gtest_filter=*{igemm_qc4w_rq_test_filter}/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}

        # Next, add test cases
        base_defaults["qc4w_scalar_test"] = {"bin": self.gemm_qc4w_test_bin, "args": f"--gtest_filter=*QD8_F32_QC4W_GEMM_MINMAX_1X4__SCALAR* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        # Add Miscellaneous defaults here
        base_defaults["rvv_gemm_test"] = {"bin": self.gemm_test_bin, "args": f"--gtest_filter=*QD8_F32_QC8W_GEMM_MINMAX_*__RVV/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["rvv_gemm_test_rq"] = {"bin": self.gemm_rq_test_bin, "args": f"--gtest_filter=*QS8_QC8W_GEMM_MINMAX*__RVV/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["rvv_igemm_test"] = {"bin": self.igemm_test_bin, "args": f"--gtest_filter=*QD8_F32_QC8W_IGEMM_MINMAX_*__RVV/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["rvv_igemm_test_rq"] = {"bin": self.igemm_rq_test_bin, "args": f"--gtest_filter=*QS8_QC8W_IGEMM_MINMAX*__RVV/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["rvv_bmm_test"] = {"bin": self.bmm_test_bin, "args": f"--gtest_filter=*BatchMatrixMultiplyQD8F32* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}


        base_defaults["imi_gemm_test"] = {"bin": self.gemm_test_bin, "args": f"--gtest_filter=*QD8_F32_QC8W_GEMM_MINMAX_*__IMI*/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["imi_gemm_test_rq"] = {"bin": self.gemm_rq_test_bin, "args": f"--gtest_filter=*QS8_QC8W_GEMM_MINMAX*__IMI*/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["imi_igemm_test"] = {"bin": self.igemm_test_bin, "args": f"--gtest_filter=*QD8_F32_QC8W_IGEMM_MINMAX_*__IMI*/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["imi_igemm_test_rq"] = {"bin": self.igemm_rq_test_bin, "args": f"--gtest_filter=*QS8_QC8W_IGEMM_MINMAX*__IMI*/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}

        ## Q4
        base_defaults["imi_gemm_test_qc4w"] = {"bin": self.gemm_test_bin, "args": f"--gtest_filter=*QD8_F32_QC4W_GEMM_MINMAX_*__IMI*/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["imi_gemm_test_rq_qc4w"] = {"bin": self.gemm_rq_test_bin, "args": f"--gtest_filter=*QS8_QC4W_GEMM_MINMAX*__IMI*/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["imi_igemm_test_qc4w"] = {"bin": self.igemm_test_bin, "args": f"--gtest_filter=*QD8_F32_QC4W_IGEMM_MINMAX_*__IMI*/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["imi_igemm_test_rq_qc4w"] = {"bin": self.igemm_rq_test_bin, "args": f"--gtest_filter=*QS8_QC4W_IGEMM_MINMAX*__IMI*/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}

        base_defaults["rvv_vlen128_gemm_test"] = {"bin": self.gemm_test_bin, "args": f"--gtest_filter=*QD8_F32_QC8W_GEMM_MINMAX_*__RVV_VLEN128/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["rvv_vlen128_gemm_test_rq"] = {"bin": self.gemm_rq_test_bin, "args": f"--gtest_filter=*QS8_QC8W_GEMM_MINMAX*__RVV_VLEN128/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["rvv_vlen128_igemm_test"] = {"bin": self.igemm_test_bin, "args": f"--gtest_filter=*QD8_F32_QC8W_IGEMM_MINMAX_*__RVV_VLEN128/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}
        base_defaults["rvv_vlen128_igemm_test_rq"] = {"bin": self.igemm_rq_test_bin, "args": f"--gtest_filter=*QS8_QC8W_IGEMM_MINMAX*__RVV_VLEN128/GemmTest* --gtest_repeat=1 --gtest_death_test_style=threadsafe --workers=1"}


        base_defaults["generate_igemm"] = {"bin": f"{self.root}/scripts/generate-qs8-igemm.sh", "args": ""}
        base_defaults["generate_gemm"] = {"bin": f"{self.root}/scripts/generate-qs8-gemm.sh", "args": ""}


        return base_defaults
        

@register_class("xnnpack_imi")
class XNNPACKIMI(XNNPACKBase):
    def __init__(self):
        super().__init__(
            "xnnpack_imi",
            "linux-build-imi",
            "xnnpack-install-imi",
            ["ukernels_imi.txt", "ukernels_rvv.txt"],
            env=IMI_ENV
        )
@register_class("xnnpack_x86")
class XNNPACKX86(XNNPACKBase):
    def __init__(self):
        super().__init__(
            "xnnpack_x86",
            "linux-build-x86",
            "xnnpack-install-x86",
            ["ukernels_cascadelake.txt"],
            use_qemu=False
        )

@register_class("xnnpack_amx")
class XNNPACKAMX(XNNPACKBase):
    def __init__(self):
        super().__init__(
            "xnnpack_amx",
            "linux-build-amx",
            "xnnpack-install-amx",
            ["ukernels_emeraldrapids.txt"],
            use_qemu=False,
            remote_info=AMX_REMOTE,
            env=AMX_ENV
        )

@register_class("xnnpack_neoverse")
class XNNPACKNeoverse(XNNPACKBase):
    def __init__(self):
        super().__init__(
            "xnnpack_neoverse",
            "linux-build-neoverse-n2",
            "xnnpack-install-neoverse-n2",
            ["ukernels_neoverse_n2.txt"],
            use_qemu=False,
            remote_info=NEOVERSE_REMOTE,
            env=NEOVERSE_ENV
        )

@register_class("xnnpack_rvv")
class XNNPACKRVV(XNNPACKBase):
    def __init__(self):
        super().__init__(
            "xnnpack_rvv",
            "linux-build-rvv",
            "xnnpack-install-rvv",
            ["ukernels_rvv.txt"],
            env=RVV_ENV
        )
