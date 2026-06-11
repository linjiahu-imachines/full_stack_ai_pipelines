from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import shutil, tarfile
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import os
from .utils import shell, get_cycles, get_perf_in_ns
from .log_cfg import logger, set_log_file
from .constants import BASE_ENV, DEV_ENV_ROOT, PERMAFROST_BIN, \
        RESULTS_DIR, QEMU_USER_BIN, QEMU_SYS_BIN, QEMU_USER_DIR, QEMU_SYS_DIR, \
        DEBUG_QEMU, BENCH_FILE, IMI_RISCV_ARCH, DEV_SCRIPT_PATHS, SPIKE_DIR, IMI_CPU_ALIAS
from .ssh import RemoteConn
from .trace_gen import generate_profile

SPIKE_IMI_RISCV_ARCH=IMI_RISCV_ARCH.replace("_ximimce", "")
QEMU_ROI_ENV_FLAG=f"-E IMI_ROI_SIM=\"1\""
PRINT_SYMBOLS=False

# Automatically enabled:
# perf::pilos::enable_prefetcher=true
# perf::pilos::gs_region_size=4096
# perf::pilos::train_prefetcher_at_LT=true
# perf::pilos::prefetch_based_on_load_pressure=true
# perf::pilos::prefetch_to_l2_on_load_pressure=true
# perf::pilos::num_prefetches_gspf=0,0,8,16
# perf::pilos::num_prefetches_cspf=0,0,4,8

PERMAFROST_SIM_CFG = {
    # From binary_cfg 
    "live_trace": f"func::live_trace={QEMU_USER_DIR}/bin/qemu-riscv64.so",
    "memory": f"func::memory={QEMU_USER_DIR}/bin/qemu-riscv64.so",
    "execution": f"func::execution={SPIKE_DIR}/build/libriscv.so",
    "argv": f"func::argv=qemu.so {QEMU_ROI_ENV_FLAG} -one-insn-per-tb -d nochain -umode -internal-syscall -cosim -plugin $carbon<COSIM_LIB> -cpu {IMI_CPU_ALIAS} $carbon<SIM_BIN> $SIM_ARGS$",
    "imi_spike": f"func::imi_spike::no_args::m_pk=--isa={SPIKE_IMI_RISCV_ARCH} $carbon<PK_BIN> $carbon<SIM_BIN> $SIM_ARGS$",
    "perf": f"perf={DEV_ENV_ROOT}/Pilos/build/libpilos.so",
    "knob_apply_umode": "perf::pilos::sch::knob_apply_umode=true",
    "threads": "perf::pilos::threads=1",
    "knob_VLEN": "perf::pilos::sch::knob_VLEN=128",
    "knob_mem_load_pipe_enabled": "perf::pilos::sch::knob_mem_load_pipe_enabled=true",
    "knob_trace_skip_vector_exe": "perf::pilos::sch::knob_trace_skip_vector_exe=true",
    "knob_max_broadcasts": "perf::pilos::knob_max_broadcasts=2",
    "knob_rom_threshold": "perf::pilos::knob_rom_threshold=4",
    "knob_enable_retire_file": "perf::pilos::sch::knob_enable_retire_file=$RETIRE_KEEP$",
    "fe_clk_ps": "perf::pilos::fe_clk_ps=1000",
    "log_start_cycle": "perf::pilos::log_start_cycle=0",
    "log_end_cycle": "perf::pilos::log_end_cycle=0 ",
    "knob_enable_exe_trace": "perf::pilos::sch::knob_enable_exe_trace=true",
    "live_trace_translation": "live_trace_translation=false",
    "allocate_partial_allowed": "perf::pilos::sch::allocate_partial_allowed=false",
    "allocate_per_rs_limits": "perf::pilos::sch::allocate_per_rs_limits=true",
    "allocate_block_on_any_full": "perf::pilos::sch::allocate_block_on_any_full=false",
    "ports": "perf::pilos::sch::allocate_pipe::ports=12",
    "generate_miss_stats": "perf::pilos::generate_miss_stats=true",
    "mc_delay": "perf::pilos::mc_delay=200",
    "allocate_vec_limit": "perf::pilos::sch::allocate_vec_limit=8",
    "allocate_int_limit": "perf::pilos::sch::allocate_int_limit=8",
    "knob_vec_prf_entries": "perf::pilos::sch::knob_vec_prf_entries=224",
    "knob_rom_throughput": "perf::pilos::knob_rom_throughput=true",
    "mem_0": "perf::pilos::sch::dist_rs::mem::0 = LD_FUN | ST_FUN",
    "mem_1": "perf::pilos::sch::dist_rs::mem::1 = LD_FUN",
    "mem_2": "perf::pilos::sch::dist_rs::mem::2 = LD_FUN",
    "mem_3": "perf::pilos::sch::dist_rs::mem::3 = LD_FUN",
    "mem_4": "perf::pilos::sch::dist_rs::mem::4 = SD",
    "mem_5": "perf::pilos::sch::dist_rs::mem::5 = ST_FUN",
    "knob_max_ab_uop_insert": "perf::pilos::knob_max_ab_uop_insert=12",
    "live_trace_offset": "live_trace_offset=31920",
}



class BaseRunner(ABC):

    @property
    def shell_build(self) -> bool:
        return False

    @property
    def upstream_remote(self) -> Optional[str]:
        return None

    @property
    def target(self) -> str:
        raise NotImplementedError("target not defined")
    
    @property
    def root(self) -> Path:
        raise NotImplementedError("root not defined")

    @property
    def root_str(self):
        return self.root if isinstance(self.root, str) else str(self.root)

    @property
    def env(self) -> Dict[str, str]:
        return {}

    @property
    def deps(self) -> List[str]:
        return []

    @property
    def shell(self) -> bool:
        return False

    @property
    def custom_scripts(self) -> List[str]:
        return []

    @property
    def ignored_submodules(self) -> Optional[List[str]]:
        return None

    @property
    def pull_cmd(self) -> Optional[List[str]]:
        return None

    def get_build_cmd(self, **kwargs) -> list:
        raise NotImplementedError("build_cmd not implemented")

    @property
    def rebuild_cmd(self) -> list:
        raise NotImplementedError("rebuild_cmd not defined")

    @property
    def default_bin(self) -> Optional[str]:
        return None

    @property
    def bin_name(self) -> Optional[str]:
        if isinstance(self.default_bin, str):
            return Path(self.default_bin).stem
        else:
            assert isinstance(self.default_bin, Path)
            return self.default_bin.stem

    @property
    def use_qemu(self) -> bool:
        return True

    @property
    def remote_info(self) -> Optional[Dict]:
        return None

    @property
    def default_runs(self) -> Dict:
        return {}
    
    def list_defaults(self, names_only: bool = False):
        if names_only:
            logger.info("NAME")
        else:
            logger.info("NAME\tCOMMAND")
        for alias, cmd in self.default_runs.items():
            if names_only:
                logger.info(f"{alias}")
            else:
                if Path(cmd['bin']).is_absolute():
                    path_str = f"{Path(cmd['bin']).relative_to(DEV_ENV_ROOT)}"
                else:
                    path_str = cmd['bin']

                logger.info(f"{alias}\t{path_str} {cmd['args']}")
    
    @property
    def post_init(self) -> List[str]:
        return []

    def check_deps(self, is_build: bool):
        return

    def validate(self) -> bool:
        raise NotImplementedError(f"Validation of working {self.target} is not yet implemented.")

    def pull(self, **kwargs):
        raise NotImplementedError

    def checkout(self, **kwargs):
        raise NotImplementedError

    def init(self, **kwargs):
        raise NotImplementedError

    def get_sim_results(self, sim_dir: Path, return_perf_info = False):
        raise NotImplementedError(f"{self.target} is not a valid target for retrieving simulation results.")

    def build(self, **kwargs):
        raise NotImplementedError

    def sync(self, **kwargs):
        raise NotImplementedError

    def rebuild(self, **kwargs):
        raise NotImplementedError

    def run(self, **kwargs):
        raise NotImplementedError

    def simulate(self, **kwargs):
        raise NotImplementedError

    def simpoint(self, **kwargs):
        raise NotImplementedError

    def sweep_sim(self, **kwargs):
        raise NotImplementedError
    
    def get_default_run(self, name: str) -> Dict:
        assert name in self.default_runs, f"{name} is not a valid default run for {self.target}"

    def exec_cmd(self, cmd: str, **kwargs):
        if cmd == "pull":
            self.pull()
        elif cmd == "checkout":
            self.checkout(**kwargs)
        elif cmd == "defaults":
            self.list_defaults(**kwargs)
        elif cmd == "init":
            self.init(**kwargs)
        elif cmd == "build":
            self.build(**kwargs)
        elif cmd == "rebuild":
            self.rebuild()
        elif cmd == "run":
            self.run(**kwargs)
        elif cmd == "simpoint":
            self.simpoint(**kwargs)
        elif cmd == "sim":
            self.simulate(**kwargs)
        elif cmd == "sync":
            self.sync(**kwargs)
        elif cmd == "sweep":
            self.sweep_sim(**kwargs)
        else:
            raise ValueError(f"{cmd} is not a valid command!")


class BenchSimRunner(BaseRunner):

    @property
    def ukernels(self) -> List[str]:
        return []

    @property
    def sweep_bin(self) -> Optional[str]:
        return None

    @property
    def git_tag(self) -> Optional[str]:
        return None

    @property
    def remote_path(self):
        raise NotImplementedError("remote_path not implemented")

    @property
    def is_git(self) -> bool:
        return "github.com" in self.remote_path

    @property
    def is_tgz(self) -> bool:
        return ".tgz" in self.remote_path

    @property
    def install_dir(self) -> Path:
        raise NotImplementedError("install_dir not implemented")

    @property
    def build_dir(self) -> Path:
        raise NotImplementedError("build_dir not implemented")

    def sync(self):
        if not self.upstream_remote or not self.is_git:
            logger.info(f"Target {self.target} does not have a remote to be synced")
            return

        if not self.root.exists():
            raise RuntimeError(f"{self.target} has not been initialized. Please run `iminnt -t {self.target} init`, then try again.")
        # Check and make sure the upstream variant has been added
        remotes = shell(f"git -C {self.root} remote".split())["msg"].split()
        if "upstream" not in remotes:
            raise RuntimeError(f"Upstream repository not setup for {self.target}.")

        # Store current branch so we can switch back later
        cur_branch_out = shell(f"git -C {self.root} symbolic-ref --short HEAD".split())
        if cur_branch_out["exit_code"] != 0:
            raise RuntimeError(f"Unable to get current branch for {self.target}: {cur_branch_out['msg']}")
        cur_branch = cur_branch_out['msg'].strip()

        branch_list = shell(f"git -C {self.root} branch --list".split())["msg"].split()
        if "upstream" not in branch_list:
            raise RuntimeError(f"Designated 'upstream' branch has not been set up for {self.target}")

        ## Do the syncing
        logger.info(f"Syncing branch 'upstream' for target {self.target}")
        # Checkout branch to be synced
        co_upstream_cmd = ["git", "-C", f"{self.root}", "checkout", "upstream"]
        shell(co_upstream_cmd)
        # Fetch upstream changes
        shell(f"git -C {self.root} fetch upstream".split())
        # Get default branch
        default_branch_out = shell(f"git -C {self.root} remote show upstream".split())["msg"]
        s = re.search("HEAD branch: (.*)", default_branch_out)
        if s is None:
            raise RuntimeError(f"Unable to obtain default branch for upstream repo in {self.target}")
        def_branch = s.group(1).strip()

        # Rebase
        shell(f"git -C {self.root} rebase upstream/{def_branch}".split())
        
        # Push
        shell(f"git -C {self.root} push -f origin upstream".split())

        ## end syncing
        logger.info(f"Successfully synced 'upstream' for target {self.target}")

        # Switch back to the original branch
        checkout_cmd = ["git", "-C", f"{self.root}", "checkout", cur_branch]
        shell(checkout_cmd)

        
    # TODO: Clean up this function, it is an awful implementation
    def init(self, reinit: bool = False, extras: bool = False):
        if reinit:
            shutil.rmtree(self.root_str)
        self.check_deps(False)

        if self.root.exists() and not extras:
            if self.is_git and not extras:
                has_changes = bool(shell(["git", "-C", f"{self.root}", "status", "-uno", "--porcelain"])['msg'].strip()) or False
                if reinit and has_changes:
                    logger.info(f"Target {self.target} has uncommitted changes and cannot be forcibly reset. Please commit the changes before force resetting")
                    return
                elif reinit:
                    pass
                else:
                    logger.info(f"Target {self.target} already initialized to {self.root}")
                    return
            else:
                return
        if extras:
            logger.info(f"Copying extra scripts for {self.target} to {self.root}")

            for cs in self.custom_scripts:
                shutil.copy(DEV_SCRIPT_PATHS / cs, str(self.root / cs))
            return
        elif self.is_git:
            logger.info(f"Cloning code for {self.target} from {self.remote_path}")

            # Clone repository
            shell(["git", "clone", "--progress", "--filter=blob:none", self.remote_path, f"{self.root}"])["msg"].strip()
            shell(["git", "-C", f"{self.root}", "submodule", "update", "--init", "--progress"])
            if self.git_tag:
                assert isinstance(self.git_tag, str)
                shell(["git", "-C", f"{self.root}", "checkout", self.git_tag])
        elif self.is_tgz:
            logger.info(f"Retrieving compressed code {self.target} from {self.remote_path}")
            assert Path(self.remote_path).exists(), f"Unable to find the remote path {self.remote_path} for target {self.target} to copy locally. Ask your administrator for access, or manually copy the source code from elsewhere instead."

            Path(self.root).mkdir(parents=True, exist_ok=True)
            tgz_path = f"{self.root}.tgz"
            shutil.copyfile(str(Path(self.remote_path)), str(tgz_path))
            with tarfile.open(tgz_path, 'r:gz') as tar:
                tar.extractall(self.root_str)
            logger.info(f"Successfully extracted '{tgz_path}' to '{self.root}'")
        elif not extras:
            logger.info(f"Retrieving code for {self.target} from {self.remote_path}")
            assert Path(self.remote_path).exists(), f"Unable to find the remote path for target {self.target} to copy locally. Ask your administrator for access, or manually copy the source code from elsewhere instead."
            shutil.copytree(str(Path(self.remote_path)), str(self.root))

        for cs in self.custom_scripts:
            assert (DEV_SCRIPT_PATHS / cs).exists()
            shutil.copy(DEV_SCRIPT_PATHS / cs, str(self.root / cs))
        
        if self.pull_cmd:
            self.pull()

        env = dict(BASE_ENV, **self.env)
        for pi in self.post_init:
            shell(pi.split(), env=env, cwd=self.root_str)


    def check_exists(self, init_on_fail: bool = True):
        assert isinstance(self.root, Path), f"Root value should be Path, not {type(self.root)}"
        if not self.root.exists() and init_on_fail:
            logger.info(f"Code for {self.target} does not yet exist locally. Proceeding to setup")
            self.init()
        elif not self.root.exists():
            logger.info(f"Code for {self.target} does not exist locally. Please initialize first")
        return self.root.exists()

    def pull(self):
        self.check_exists()
        # TODO: add assertion to check for valid target
        if self.pull_cmd:
            env = dict(BASE_ENV, **self.env)
            for pcmd in self.pull_cmd:
                if self.shell:
                    shell(pcmd, shell=True, env=env, cwd=self.root_str)
                else:
                    shell(pcmd.split(), env=env, cwd=self.root_str)
        else:
            pull_cmd = ["git", "-C", f"{self.root}", "pull"]
            submod_cmd = ["git", "-C", f"{self.root}", "submodule", "update", "--recursive"]
            shell(pull_cmd)
            shell(submod_cmd)
    
    def checkout(self, branch: str):
        if not self.is_git:
            raise RuntimeError(f"{self.target} is not a git repository. Unable to run `git checkout`.")
        logger.info(f"Checking out branch {branch} in for {self.target}")
        checkout_cmd = ["git", "-C", f"{self.root}", "checkout", branch]
        shell(checkout_cmd)

    def build(self, threads: Optional[int] = None):
        if threads is not None:
            assert isinstance(threads, int) and threads >= 1, f"Thread count for building must be at least 1"
        self.check_deps(True)
        self.check_exists()
        env = dict(BASE_ENV, **self.env)
        build_cmd = self.get_build_cmd(threads=threads)
        assert isinstance(build_cmd, list)
        for bcmd in build_cmd:
            if self.shell_build:
                shell(bcmd, shell=True, env=env, cwd=self.root_str)
            else:
                shell(bcmd.split(), env=env, cwd=self.root_str)

    def _get_run_cmd(self, bin_name: str = None, default_cmd: Optional[str] = None, bin_args: Optional[List[str]] = None):
        cmd_info = {}
        if self.default_bin is None:
            raise ValueError(f"Expected default binary or alternative binary specified for target {self.target}")

        if default_cmd:
            assert default_cmd in self.default_runs, f"{default_cmd} is not a valid default command for target {self.target}. Options are {list(self.default_runs.keys())}"
            logger.info(f"Using default command {default_cmd}, overriding other arguments.")
            di = self.default_runs[default_cmd]
            bin_name = di["bin"]
            bin_path = Path(self.root_str) / bin_name
            assert bin_path.exists(), f"No such binary executable at path {bin_path}"
            bin_args_str = di.get("args", "")
            bin_args_list = shlex.split(bin_args_str) if bin_args_str else []
            run_env = dict(BASE_ENV, **di.get("env", {}))
        else:
            bin_name = bin_name or self.default_bin
            if Path(bin_name).is_absolute():
                bin_path = Path(bin_name)
                logger.info(f"Using absolute path for binary at {bin_path}")
            elif (self.install_dir / bin_name).exists():
                bin_path = self.install_dir / bin_name
                logger.info(f"Using binary located in install directory at {bin_path}")
            elif (self.build_dir / bin_name).exists():
                bin_path = self.build_dir / bin_name
                logger.info(f"Using binary located in build directory at {bin_path}")
            elif (self.root / bin_name).exists():
                bin_path = self.root / bin_name
                logger.info(f"Using binary in root directory at {bin_path}")
            else:
                raise ValueError(f"Unable to locate binary {bin_name}. Binaries passed as command line arguments must be either absolute paths, or be relative paths to the install or build directories for {self.target}")
            assert bin_path.exists(), f"No such binary executable at path {bin_path}"
            if isinstance(bin_args, list) and len(bin_args) > 0:
                bin_args_list = bin_args
                bin_args_str = " ".join(bin_args)
            else:
                bin_args_list = []
                bin_args_str = ""
            run_env = dict(BASE_ENV, **self.env)
        bin_cmd = f"{bin_path} {bin_args_str}" if bin_args_str else str(bin_path)
        cmd_info["env"] = run_env
        cmd_info["cmd"] = bin_cmd
        cmd_info["bin_name"] = bin_name
        cmd_info["bin_path"] = bin_path
        cmd_info["bin_args"] = bin_args_str
        cmd_info["bin_args_list"] = bin_args_list
        return cmd_info

    def get_offsets(self, bin_name: str = None, default_cmd: Optional[str] = None, bin_args: Optional[List[str]] = None):
        cmd_info = self._get_run_cmd(bin_name, default_cmd, bin_args)
        run_env = cmd_info["env"]
        bin_name = cmd_info["bin_name"]
        bin_cmd = cmd_info["cmd"]
        qemu_bin = f"{QEMU_USER_BIN} -cpu {IMI_CPU_ALIAS} -plugin {QEMU_USER_DIR}/plugins/libprint-roi-region.so"
        qemu_cmd = f"{qemu_bin} {bin_cmd}"
        msg = shell(qemu_cmd.split(), silent=True)["msg"]
        msg_lines = msg.splitlines()
        starts = []
        ends = []
        for l in msg_lines:
            if "csr: csrrsi" in l:
                s = re.search("instruction offset: (\d+)", l)
                assert s is not None
                starts.append(int(s.group(1)))
            elif "csr: csrrci" in l:
                e = re.search("instruction offset: (\d+)", l)
                assert e is not None
                ends.append(int(e.group(1)))
        starts = [int(s) for s in starts]
        ends = [int(e) for e in ends]
        assert len(starts) > 0, f"No start offsets found"
        assert len(ends) > 0, f"No end offsets found"
        min_start = min(starts)
        max_end = min(ends)
        assert min_start < max_end, f"Minimum start instruction is larger than end instruction"
        return min_start, max_end

    def get_program_start_addr(self, bin_name: str = None, default_cmd: Optional[str] = None, bin_args: Optional[List[str]] = None):
        ADDR_STR_PREFIX = "[plugin] Detected main() at"
        cmd_info = self._get_run_cmd(bin_name, default_cmd, bin_args)
        run_env = cmd_info["env"]
        bin_name = cmd_info["bin_name"]
        bin_cmd = cmd_info["cmd"]
        qemu_bin = f"{QEMU_USER_BIN} -cpu {IMI_CPU_ALIAS} -plugin {QEMU_USER_DIR}/plugins/libpstart_finder.so"
        qemu_cmd = f"{qemu_bin} {bin_cmd}"
        msg = shell(qemu_cmd.split(), silent=False)["msg"]
        msg_lines = msg.splitlines()
        starts = []
        for l in msg_lines:
            if ADDR_STR_PREFIX in l:
                logger.info(f"Found program start in line: {l}")
                s = re.search(f"(0x\w+)", l)
                assert s is not None
                starts.append(int(s.group(1), 16))
        if len(starts) > 1:
            raise RuntimeError(f"Found multiple start addresses, expected 1: {starts}")
        elif len(starts) == 0:
            raise RuntimeError(f"Unable to identify program start address")
        return starts[0]

    def get_program_icount(self, bin_name: str = None, default_cmd: Optional[str] = None, bin_args: Optional[List[str]] = None):
        ICOUNT_STR_PREFIX = "QEMU:icount:  Executed"
        cmd_info = self._get_run_cmd(bin_name, default_cmd, bin_args)
        run_env = cmd_info["env"]
        bin_name = cmd_info["bin_name"]
        bin_cmd = cmd_info["cmd"]
        qemu_bin = f"{QEMU_USER_BIN} -cpu {IMI_CPU_ALIAS} -plugin {QEMU_USER_DIR}/plugins/libicount-roi.so"
        qemu_cmd = f"{qemu_bin} {bin_cmd}"
        msg = shell(qemu_cmd.split(), silent=False)["msg"]
        msg_lines = msg.splitlines()
        icounts = []
        for l in msg_lines:
            if ICOUNT_STR_PREFIX in l:
                s = re.search(f"(\d+) instructions", l)
                assert s is not None
                icounts.append(int(s.group(1)))
        if len(icounts) > 1:
            raise RuntimeError(f"Found instruction counts, expected 1: {icounts}")
        elif len(icounts) == 0:
            raise RuntimeError(f"Unable to calculate number of instructions")
        return icounts[0]

    def rebuild(self):
        self.check_exists(False)
        env = dict(BASE_ENV, **self.env)
        assert isinstance(self.rebuild_cmd, list)
        for rbcmd in self.rebuild_cmd:
            if self.shell:
                shell(rbcmd, shell=True, env=env, cwd=self.root_str)
            else:
                shell(rbcmd.split(), env=env, cwd=self.root_str)

    def is_static_bin(self, bin_path: str) -> bool:
        res = shell(["file", bin_path], no_fail=True, silent=True)['msg']
        return "statically linked" in res

    def _resolve_path_relative_to_root(self, path: Path) -> Path:
        """Resolve a path that might be relative to project root"""
        if path.is_absolute() and path.exists():
            return path
        
        # Try relative to root
        root_path = Path(self.root_str) if hasattr(self, 'root_str') else Path(str(self.root))
        path_clean = str(path).lstrip('/')
        resolved = root_path / path_clean
        if resolved.exists():
            return resolved
        
        # Try PROMPTS_DIR for prompt files
        if "--file" in str(path) or path.suffix in ['.txt', '.md']:
            from .constants import PROMPTS_DIR
            if (PROMPTS_DIR / path_clean).exists():
                return PROMPTS_DIR / path_clean
        
        # Try resolving from current directory
        try:
            resolved = path.resolve()
            if resolved.exists():
                return resolved
        except:
            pass
        
        return path  # Return as-is if can't resolve

    def _prepare_system_mode_shared_dir(self, bin_path: Path, bin_args: str, smp_cores: int):
        """Prepare shared directory for 9p virtfs: copy binary, models, and create init script.
        Returns: (shared_dir_path, updated_args, init_script_content)"""
        import tempfile
        import shutil
        
        # Update thread count in args to match cores
        updated_args = bin_args
        if "-t " in updated_args:
            updated_args = re.sub(r'-t\s+\d+', f'-t {smp_cores}', updated_args)
        else:
            if "-m " in updated_args:
                updated_args = re.sub(r'(-m\s+\S+)', rf'\1 -t {smp_cores}', updated_args, count=1)
            else:
                updated_args = f"-t {smp_cores} {updated_args}"
        
        # Create temporary shared directory (will be mounted via 9p)
        shared_dir = tempfile.mkdtemp(prefix="iminn_shared_")
        shared_path = Path(shared_dir)
        
        # Copy binary
        bin_dir = shared_path / "bin"
        bin_dir.mkdir(exist_ok=True)
        dest_bin = bin_dir / bin_path.name
        shutil.copy2(bin_path, dest_bin)
        dest_bin.chmod(0o755)
        
        # Copy models if they exist
        if "-m " in updated_args:
            model_match = re.search(r'-m\s+(\S+)', updated_args)
            if model_match:
                model_path_str = model_match.group(1)
                model_path = self._resolve_path_relative_to_root(Path(model_path_str))
                
                if model_path.exists():
                    models_dir = shared_path / "models"
                    models_dir.mkdir(exist_ok=True)
                    shutil.copy2(model_path, models_dir / model_path.name)
                    # Update args to use /mnt/iminn/models path
                    updated_args = re.sub(r'-m\s+' + re.escape(model_path_str), f'-m /mnt/iminn/models/{model_path.name}', updated_args)
                else:
                    logger.warning(f"Model file not found: {model_path}, skipping model copy")
        
        # Copy prompt files if referenced
        if "--file " in updated_args:
            file_match = re.search(r'--file\s+(\S+)', updated_args)
            if file_match:
                prompt_path_str = file_match.group(1)
                prompt_path = self._resolve_path_relative_to_root(Path(prompt_path_str))
                
                if prompt_path.exists():
                    prompts_dir = shared_path / "prompts"
                    prompts_dir.mkdir(exist_ok=True)
                    shutil.copy2(prompt_path, prompts_dir / prompt_path.name)
                    # Update args to use /mnt/iminn/prompts path
                    updated_args = re.sub(r'--file\s+' + re.escape(prompt_path_str), f'--file /mnt/iminn/prompts/{prompt_path.name}', updated_args)
                else:
                    logger.warning(f"Prompt file not found: {prompt_path}, skipping prompt copy")
        
        # Create init script content - will be executed from rootfs
        init_content = f"""#!/bin/sh
# Mount required filesystems
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true

# Create necessary directories
mkdir -p /tmp /run /var/run /mnt/iminn

# Mount 9p shared directory
if ! mount -t 9p -o trans=virtio,version=9p2000.L iminn_share /mnt/iminn; then
    echo "ERROR: Could not mount 9p filesystem"
    echo "Available 9p devices:"
    ls -la /dev/9p* 2>/dev/null || true
    echo "Sleeping for debugging..."
    sleep 3600
    exit 1
fi

# Run the command from shared directory
echo "=========================================="
echo "Running: /mnt/iminn/bin/{bin_path.name} {updated_args}"
echo "=========================================="
/mnt/iminn/bin/{bin_path.name} {updated_args}
EXIT_CODE=$?

# Power off when done
echo "=========================================="
echo "Command completed with exit code: $EXIT_CODE"
echo "=========================================="
poweroff -f 2>/dev/null || /sbin/poweroff -f 2>/dev/null || /bin/poweroff -f 2>/dev/null || echo "Please power off manually"
"""
        
        # Write init script to shared dir (will be accessible via 9p)
        # Note: shared_path is already defined above at line 605
        init_script = shared_path / "init.sh"
        with open(init_script, 'w') as f:
            f.write(init_content)
        init_script.chmod(0o755)
        
        return shared_dir, updated_args, init_content

    def _prepare_system_mode_initrd(self, bin_path: Path, bin_args: str, smp_cores: int) -> tuple[Path, str]:
        """Create a complete initrd (cpio archive) with binary, models, prompts, and init script.
        Returns: (initrd_path, updated_args)"""
        import tempfile
        import subprocess
        import shutil
        
        # Update thread count in args to match cores
        updated_args = bin_args
        if "-t " in updated_args:
            updated_args = re.sub(r'-t\s+\d+', f'-t {smp_cores}', updated_args)
        else:
            if "-m " in updated_args:
                updated_args = re.sub(r'(-m\s+\S+)', rf'\1 -t {smp_cores}', updated_args, count=1)
            else:
                updated_args = f"-t {smp_cores} {updated_args}"
        
        # Create temporary directory for initrd
        initrd_dir = tempfile.mkdtemp(prefix="iminn_initrd_")
        initrd_path_obj = Path(initrd_dir)
        
        try:
            # Create directory structure
            (initrd_path_obj / "bin").mkdir(exist_ok=True)
            (initrd_path_obj / "models").mkdir(exist_ok=True)
            (initrd_path_obj / "prompts").mkdir(exist_ok=True)
            (initrd_path_obj / "proc").mkdir(exist_ok=True)
            (initrd_path_obj / "sys").mkdir(exist_ok=True)
            (initrd_path_obj / "dev").mkdir(exist_ok=True)
            (initrd_path_obj / "tmp").mkdir(exist_ok=True)
            
            # Copy binary
            dest_bin = initrd_path_obj / "bin" / bin_path.name
            shutil.copy2(bin_path, dest_bin)
            dest_bin.chmod(0o755)
            
            # Copy models if they exist
            if "-m " in updated_args:
                model_match = re.search(r'-m\s+(\S+)', updated_args)
                if model_match:
                    model_path_str = model_match.group(1)
                    model_path = self._resolve_path_relative_to_root(Path(model_path_str))
                    
                    if model_path.exists():
                        shutil.copy2(model_path, initrd_path_obj / "models" / model_path.name)
                        # Update args to use /models path in initrd
                        updated_args = re.sub(r'-m\s+' + re.escape(model_path_str), f'-m /models/{model_path.name}', updated_args)
                    else:
                        logger.warning(f"Model file not found: {model_path}, skipping model copy")
            
            # Copy prompt files if referenced
            if "--file " in updated_args:
                file_match = re.search(r'--file\s+(\S+)', updated_args)
                if file_match:
                    prompt_path_str = file_match.group(1)
                    prompt_path = self._resolve_path_relative_to_root(Path(prompt_path_str))
                    
                    if prompt_path.exists():
                        shutil.copy2(prompt_path, initrd_path_obj / "prompts" / prompt_path.name)
                        # Update args to use /prompts path in initrd
                        updated_args = re.sub(r'--file\s+' + re.escape(prompt_path_str), f'--file /prompts/{prompt_path.name}', updated_args)
                    else:
                        logger.warning(f"Prompt file not found: {prompt_path}, skipping prompt copy")
            
            # Create init script
            init_content = f"""#!/bin/sh
# Mount essential filesystems
mount -t proc proc /proc
mount -t sysfs sysfs /sys
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
mount -t tmpfs tmpfs /tmp 2>/dev/null || true

# Set PATH
export PATH=/bin:/usr/bin

# Run the command
echo "=========================================="
echo "Running: /bin/{bin_path.name} {updated_args}"
echo "=========================================="
/bin/{bin_path.name} {updated_args}
EXIT_CODE=$?

# Power off when done
echo "=========================================="
echo "Command completed with exit code: $EXIT_CODE"
echo "=========================================="
poweroff -f 2>/dev/null || /sbin/poweroff -f 2>/dev/null || /bin/poweroff -f 2>/dev/null || echo "Please power off manually"
"""
            
            init_file = initrd_path_obj / "init"
            with open(init_file, 'w') as f:
                f.write(init_content)
            init_file.chmod(0o755)
            
            # Create initrd cpio archive
            initrd_file = tempfile.NamedTemporaryFile(prefix="iminn_initrd_", suffix=".cpio", delete=False)
            initrd_file.close()
            initrd_path = Path(initrd_file.name)
            
            # Create cpio archive
            result = subprocess.run(
                ["sh", "-c", f"cd {initrd_dir} && find . | cpio -o -H newc > {initrd_path}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
            
            return initrd_path, updated_args
        finally:
            # Cleanup initrd directory
            try:
                shutil.rmtree(initrd_dir, ignore_errors=True)
            except:
                pass

    def run(self, bin_name: str = None, default_cmd: Optional[str] = None, bin_args: Optional[List[str]] = None, icount: bool = False, system_mode: bool = False, smp_cores: int = 4, kernel_path: Optional[str] = None, rootfs_path: Optional[str] = None):
        if icount and not self.use_qemu:
            raise RuntimeError(f"Cannot collect instruction count for non-riscv/qemu-based target {self.target}")
        self.check_exists(False)
        cmd_info = self._get_run_cmd(bin_name, default_cmd, bin_args)
        run_env = cmd_info["env"]
        bin_cmd = cmd_info["cmd"]
        bin_name = cmd_info["bin_name"]
        bin_path = cmd_info["bin_path"]
        bin_args = cmd_info["bin_args"]
        bin_args_list = cmd_info["bin_args_list"]
        if self.use_qemu:
            if system_mode:
                # System mode execution path using initrd (no sudo needed, no 9p issues!)
                from .constants import QEMU_SYS_BIN, QEMU_SYS_KERNEL, QEMU_SYS_MEMORY
                import atexit
                import shutil
                
                kernel = kernel_path or str(QEMU_SYS_KERNEL)
                
                # Prepare initrd with binary, models, prompts, and init script
                print(f"Preparing initrd for system mode (no sudo required)...")
                print(f"  Binary: {bin_path}")
                print(f"  Updating thread count to {smp_cores} to match CPU cores")
                initrd_path, updated_args = self._prepare_system_mode_initrd(bin_path, bin_args, smp_cores)
                
                # Register cleanup for initrd
                def cleanup_initrd():
                    try:
                        if initrd_path.exists():
                            initrd_path.unlink()
                    except:
                        pass
                atexit.register(cleanup_initrd)
                
                # Construct system mode QEMU command with initrd
                qemu_cmd = [
                    str(QEMU_SYS_BIN),
                    "-machine", "virt",
                    "-cpu", IMI_CPU_ALIAS,
                    "-smp", str(smp_cores),
                    "-m", QEMU_SYS_MEMORY,
                    "-kernel", kernel,
                    "-initrd", str(initrd_path),
                    "-append", "console=ttyS0",
                    "-nographic",
                    "-serial", "mon:stdio"
                ]
                
                print(f"Booting QEMU system mode with {smp_cores} cores...")
                print(f"Kernel: {kernel}")
                print(f"Initrd: {initrd_path}")
                print(f"Command will execute automatically from initrd")
                print(f"\n{'='*70}")
                print(f"TERMINATION INSTRUCTIONS:")
                print(f"  - Press Ctrl+C to terminate (may take a moment)")
                print(f"  - Or press Ctrl+A then 'C' to enter QEMU monitor, then type 'quit'")
                print(f"  - Or press Ctrl+A then 'X' to force quit")
                print(f"{'='*70}\n")
                
                # Use the shell function from utils which handles signals properly
                shell(qemu_cmd, env=run_env, silent=False)
            else:
                # User mode execution path
                qemu_argv = shlex.split(f"{QEMU_USER_BIN} {QEMU_ROI_ENV_FLAG} -cpu {IMI_CPU_ALIAS}")

                if not self.is_static_bin(str(bin_path)):
                    assert "CROSS_SYSROOT" in self.env
                    qemu_argv += ["-L", self.env["CROSS_SYSROOT"]]
                bin_argv = [str(bin_path)] + bin_args_list
                if PRINT_SYMBOLS:
                    qemu_argv += ["-plugin", f"{QEMU_USER_DIR}/plugins/libfn_symbol_trace.so"] + bin_argv
                elif icount:
                    qemu_argv += ["-plugin", f"{QEMU_USER_DIR}/plugins/libicount-roi.so"] + bin_argv
                else:
                    qemu_argv += bin_argv
                if icount:
                    ICOUNT_STR_PREFIX = "QEMU:icount:  Executed"
                    msg = shell(qemu_argv, env=run_env, silent=False)["msg"]
                    msg_lines = msg.splitlines()
                    icounts = []
                    for l in msg_lines:
                        if ICOUNT_STR_PREFIX in l:
                            s = re.search(f"(\d+) instructions", l)
                            assert s is not None
                            icounts.append(int(s.group(1)))
                    if len(icounts) > 1:
                        raise RuntimeError(f"Found instruction counts, expected 1: {icounts}")
                    elif len(icounts) == 0:
                        raise RuntimeError(f"Unable to calculate number of instructions")
                    print(f"*****Instruction count={icounts[0]}*********")
                else:
                    shell(qemu_argv, env=run_env)
        elif self.remote_info:
            assert "name" in self.remote_info
            assert "uname" in self.remote_info
            assert "host" in self.remote_info
            assert "port" in self.remote_info
            assert "bench_dir" in self.remote_info
            bench_dir = Path(self.remote_info["bench_dir"])
            remote_bin_path = f"{bench_dir}/{Path(bin_name).stem}"
            rconn = RemoteConn(self.remote_info["name"], self.remote_info["uname"], self.remote_info["host"], self.remote_info["port"], use_agent=self.remote_info.get("use_agent", False))
            rconn.check_dir_exists(bench_dir, fail=False)
            remote_bin_cmd = f"{remote_bin_path} {bin_args}"
            rconn.copy_to_remote(bin_path, remote_bin_path,  0o755)
            rconn.run_cmd(remote_bin_cmd)
        else:
            shell(bin_cmd.split(), env=run_env)
    
    def _check_sim_cfg(self, cfg: List[str]):
        FLAGGED_CFG_OPTS = ["func::live_trace", "func::memory", "func::execution", "perf"]
        RESERVED_VARS = ["SIM_BIN", "COSIM_LIB", "PK_BIN"]
        new_cfg = []
        # We need to make sure the binary execution is correctly defined, and doesn't overwrite the intended binary execution supplied
        # to the python CLI.
        argv_idx = 0
        imi_spike_idx = 0
        for i, l in enumerate(cfg):
            # skip comments, but preserve them
            if l.lstrip().startswith("#"):
                new_cfg.append(l)
                continue
            if "=" in l:
                key_val = l.lstrip().split("=")
                key = key_val[0]
                val = "".join(key_val[1:])
                if key in FLAGGED_CFG_OPTS:
                    raise ValueError(f"Invalid custom config. The following config options cannot be supplied by the user-supplied config: {FLAGGED_CFG_OPTS}")
                elif key == "func::argv":
                    # Prevent user-supplied cosimulation library
                    if "libcosim.so" in val:
                        raise ValueError(f"User-supplied func::argv value cannot include the path to libcosim.so plugin. Instead, use the $carbon<COSIM_LIB> variable: {l}")

                    if "$carbon<SIM_BIN>" not in val:
                        raise ValueError(f"User-supplied func::argv value must specify the binary with the '$carbon<SIM_BIN>' variable, but it was not supplied: {l}")

                    if "$SIM_ARGS$" not in val:
                        raise ValueError(f"User-supplied func::argv value must specify the binary arguments with the '$SIM_ARGS$' template variable, but it was not supplied: {l}")
                    argv_idx = i

                elif key == "func::imi_spike::no_args::m_pk":
                    # Prevent user-supplied pk library
                    if "/pk" in val:
                        raise ValueError(f"User-supplied func::imi_spike::no_args::m_pk value cannot include the path to pk binary. Instead, use the $carbon<PK_BIN> variable: {l}")

                    if "$carbon<SIM_BIN>" not in val:
                        raise ValueError(f"User-supplied func::imi_spike::no_args::m_pk value must specify the binary with the '$carbon<SIM_BIN>' variable, but it was not supplied: {l}")

                    if "$SIM_ARGS$" not in val:
                        raise ValueError(f"User-supplied func::imi_spike::no_args::m_pk value must specify the binary arguments with the '$SIM_ARGS$' template variable, but it was not supplied: {l}")
                    imi_spike_idx = i
                elif key == "-carbon_set":
                    var_name = val.split(",")[0]
                    if var_name in RESERVED_VARS:
                        raise ValueError(f"Encountered reserved variable {var_name} in `-carbon_set` config. The following variables are reserved for internal use: {RESERVED_VARS}")
            new_cfg.append(l.rstrip('\n'))

        if argv_idx == 0:
            new_cfg.insert(1, f"func::argv=qemu.so -one-insn-per-tb -d nochain -umode -internal-syscall -cosim -plugin {QEMU_USER_DIR}/plugins/libcosim.so {IMI_CPU_ALIAS} $carbon<SIM_BIN> $SIM_ARGS$")

        if imi_spike_idx == 0:
            new_cfg.insert(2, f"func::imi_spike::no_args::m_pk=--isa={SPIKE_IMI_RISCV_ARCH} {SPIKE_DIR}/pk_binary/pk $carbon<SIM_BIN> $SIM_ARGS$")
            imi_spike_idx = 2

        new_cfg.insert(imi_spike_idx+1, PERMAFROST_SIM_CFG["live_trace"])
        new_cfg.insert(imi_spike_idx+1, PERMAFROST_SIM_CFG["memory"])
        new_cfg.insert(imi_spike_idx+1, PERMAFROST_SIM_CFG["execution"])
        new_cfg.insert(imi_spike_idx+1, PERMAFROST_SIM_CFG["perf"])
        return new_cfg
    
    def _init_sim_cfg(self, bin_path: str, bin_args: str, keep_retires: bool, out_dir: Path, custom_cfg: Optional[Path] = None, print_cfg: bool = False, use_roi: bool = False) -> str:
        if custom_cfg:
            custom_cfg = Path(custom_cfg)
            if not custom_cfg.exists():
                raise ValueError(f"User-provided configuration file at path {custom_cfg} could not be found.")
            with open(str(custom_cfg), "r") as f:
                sim_cfg = f.readlines()
            init_cfg = self._check_sim_cfg(sim_cfg)
        else:
            # Create a cfg for everything
            log_cfg = str(DEV_ENV_ROOT / "Pilos" / "Configs" / "disable_logs.cfg")
            aetos_cfg = str(DEV_ENV_ROOT / "Pilos" / "Configs" / "aetos_hp.cfg")
            init_cfg = [
                f"-cfg {log_cfg}",
                f"-cfg {aetos_cfg}",
                "perf_setup_delay=0"
            ] + list(PERMAFROST_SIM_CFG.values())
            if use_roi:
                init_cfg.append("wait_for_roi=1")
                init_cfg.append("listen_to_roi=true")
            else:
                init_cfg.append("wait_for_roi=0")
                init_cfg.append("listen_to_roi=false")
                removed_roi_flag = False
                for i in range(len(init_cfg)):
                    if "func::argv=qemu.so" in init_cfg[i]:
                        assert QEMU_ROI_ENV_FLAG in init_cfg[i], f"Unable to identify ROI env specification in config"
                        init_cfg[i] = init_cfg[i].replace(QEMU_ROI_ENV_FLAG, "")
                        removed_roi_flag = True

                if not removed_roi_flag:
                    raise RuntimeError(f"Unable to remove ROI flag")

        if not self.is_static_bin(str(bin_path)):
            assert "CROSS_SYSROOT" in self.env
            linked_sysroot = False
            
            for i in range(len(init_cfg)):
                if "func::argv=qemu.so" in init_cfg[i]:
                    linked_str = f"func::argv=qemu.so -L {self.env['CROSS_SYSROOT']}"
                    init_cfg[i] = init_cfg[i].replace("func::argv=qemu.so", linked_str)
                    linked_sysroot = True
                    break

            if not linked_sysroot:
                raise RuntimeError(f"Unable to link sysroot for binary with shared libs")
    
        init_cfg.insert(0, f"-carbon_set=SIM_BIN,{bin_path}")
        init_cfg.insert(0, f"-carbon_set=COSIM_LIB,{QEMU_USER_DIR}/plugins/libcosim.so")
        init_cfg.insert(0, f"-carbon_set=PK_BIN,{SPIKE_DIR}/pk_binary/pk")
        sim_cfg = []

        # Replace variable arguments
        for e in init_cfg:
            e_cfg = e
            if "$SIM_ARGS$" in e_cfg:
                e_cfg = e_cfg.replace("$SIM_ARGS$", bin_args)
            if "$RETIRE_KEEP$" in e_cfg:
                e_cfg = e_cfg.replace("$RETIRE_KEEP$", str(keep_retires).lower())
            sim_cfg.append(e_cfg)

        sim_cfg_str = "\n".join(sim_cfg)
        if print_cfg:
            logger.info(sim_cfg_str)
        cfg_path = Path(f"{out_dir}/iminnt.cfg")
        with open(str(cfg_path), "w") as f:
            f.writelines("\n".join(sim_cfg))
        return cfg_path

    
    def simpoint(self, bin_name: str = None, default_cmd: Optional[str] = None, bin_args: Optional[List[str]] = None, output_dir: Optional[str] = None, print_cfg: bool = False, isolate_program: bool = False, keep_retires: bool = False, custom_cfg: Optional[str] = None, fn_perf_graph: bool = False):
        self.check_exists(False)
        if not (DEV_ENV_ROOT / "simpoint").exists():
            raise RuntimeError("The `simpoint` target must be initialized before simpointing. Run `iminnt -t simpoint init` to initialize.")
        if not self.use_qemu:
            raise RuntimeError(f"Target {self.target} does not support simpointing.")
        
        cmd_info = self._get_run_cmd(bin_name, default_cmd, bin_args)
        run_env = cmd_info["env"]
        bin_name = cmd_info["bin_name"]
        bin_path = cmd_info["bin_path"]
        bin_args = cmd_info["bin_args"]
        bin_cmd = cmd_info["cmd"]

        if output_dir:
            out_dir = Path(output_dir)
            out_dir = RESULTS_DIR / out_dir if not out_dir.is_absolute() else out_dir
        else:
            out_dir = RESULTS_DIR / self.target

        out_dir.mkdir(parents=True, exist_ok=True)
        set_log_file(f"{out_dir}/simpointbench.log")
        # TODO: First, we should check the number of instructions for a given program and make sure it is worth it to do simpointing rather than directly running
        # icount = self.get_program_icount(bin_name=bin_name, default_cmd=default_cmd, bin_args=bin_args)
        # logger.info(f"Number of instructions: {icount}")
        link_args = ""
        if not self.is_static_bin(str(bin_path)):
            assert "CROSS_SYSROOT" in self.env
            link_args = f"-L {self.env['CROSS_SYSROOT']}"
        if isolate_program:
            qemu_bin = f"{QEMU_USER_BIN} -cpu {IMI_CPU_ALIAS} -plugin {QEMU_USER_DIR}/plugins/libsimpbbprof.so,interval=10000000,roidefault=off,bbfile={out_dir}/qemu_simpoint.bb,bblogfile={out_dir}/BB.log {link_args}"
            pstart = self.get_program_start_addr(bin_name=bin_name, default_cmd=default_cmd, bin_args=bin_args)
            logger.info(f"Retrieved program start at address: {pstart}")
            raise NotImplementedError(f"Isolated program execution is not yet implemented")
        else:
            logger.info(f"Storing simpoint results in {out_dir}")
            # First, generate bb file
            logger.info(f"Generating bb file in {out_dir}/BB.log")
            qemu_bin = f"{QEMU_USER_BIN} -cpu {IMI_CPU_ALIAS} -plugin {QEMU_USER_DIR}/plugins/libsimpbbprof.so,interval=10000000,bbfile={out_dir}/qemu_simpoint.bb,bblogfile={out_dir}/BB.log {link_args}"
            qemu_cmd = f"{qemu_bin} {bin_cmd}"
            shell(qemu_cmd.split(), env=run_env)

        if not (out_dir / "qemu_simpoint.bb").exists():
            raise RuntimeError(f"The simpoint file was not generated for this execution!")

        # TODO: add customization options to these commands
        # Now, generate simpoints
        sp_gen_cmd = f"{DEV_ENV_ROOT}/simpoint/bin/simpoint -maxK 30 -saveSimpoints {out_dir}/simpoints2 -saveSimpointWeights {out_dir}/weights -loadFVFile {out_dir}/qemu_simpoint.bb"
        shell(sp_gen_cmd.split(), env=run_env)

        sp_run_cmd = f"python3 {DEV_ENV_ROOT}/Pilos/Tools/Scripts/run_simpoints.py"
        # Specify permafrost binary and simpoint weight/simpoints locations
        sp_run_cmd = f"{sp_run_cmd} -b {PERMAFROST_BIN} -w {out_dir}/weights -s {out_dir}"
        # Specify % coverage=100%
        sp_run_cmd = f"{sp_run_cmd} -c 100"
        # Specify number of instructions in millions 
        sp_run_cmd = f"{sp_run_cmd} -l 10"
        # Lastly, specify number of threads, write to csv, and output directory
        sp_run_cmd = f"{sp_run_cmd} --write_to_csv -o {out_dir} -t 7"

        # Create a cfg for everything
        cfg_path = self._init_sim_cfg(bin_path, bin_args, keep_retires, out_dir, custom_cfg=custom_cfg, print_cfg=print_cfg, use_roi=False)
        sp_run_cmd = f"{sp_run_cmd} -- {cfg_path}"
        shell(sp_run_cmd.split(), cwd=str(out_dir), env=run_env)

        # TODO: Retrieve cycles and timing
        # time_ns = get_cycles(out_dir, is_permafrost=self.use_qemu)
        # time_ns = list(time_ns.values())[0]
        # if time_ns is None:
        #     logger.info(f"Unable to get total nanoseconds results")
        # else:
        #     logger.info(f"Total time: {time_ns} ns")
        if fn_perf_graph:
            logger.info(f"Creating profile trace in {out_dir}/trace.json for flamegraph viewing.")
            generate_profile(out_dir, logger=logger)


    def simulate(self, bin_name: str = None, default_cmd: Optional[str] = None, bin_args: Optional[List[str]] = None, output_dir: Optional[str] = None, print_cfg: bool = False, keep_retires: bool = False, custom_cfg: Optional[str] = None, no_roi: bool = False, fn_perf_graph: bool = False):
        self.check_exists(False)
        cmd_info = self._get_run_cmd(bin_name, default_cmd, bin_args)
        run_env = cmd_info["env"]
        bin_name = cmd_info["bin_name"]
        bin_path = cmd_info["bin_path"]
        bin_args = cmd_info["bin_args"]
        bin_cmd = cmd_info["cmd"]


        # Directory where simulate is executed is where the outputs are stored
        if output_dir:
            out_dir = Path(output_dir)
            out_dir = RESULTS_DIR / out_dir if not out_dir.is_absolute() else out_dir
        else:
            out_dir = RESULTS_DIR / self.target

        out_dir.mkdir(parents=True, exist_ok=True)
        set_log_file(f"{out_dir}/simbench.log")
        logger.info(f"Storing simulation results in {out_dir}")

        if self.use_qemu:
            cfg_path = self._init_sim_cfg(bin_path, bin_args, keep_retires, out_dir, custom_cfg=custom_cfg, print_cfg=print_cfg, use_roi=not no_roi)
            sim_cmd = [str(PERMAFROST_BIN), str(cfg_path)]
            shell(sim_cmd, cwd=str(out_dir), env=run_env)
            if fn_perf_graph:
                logger.info(f"Creating profile trace in {out_dir}/trace.json for flamegraph viewing.")
                generate_profile(out_dir, logger=logger)
        elif self.remote_info:
            assert "name" in self.remote_info
            assert "uname" in self.remote_info
            assert "host" in self.remote_info
            assert "port" in self.remote_info
            assert "bench_dir" in self.remote_info
            bench_dir = Path(remote_info["bench_dir"])
            remote_bin_path = f"{bench_dir}/{Path(bin_name).stem}"
            remote_bin_args = bin_args if isinstance(bin_args, str) else " ".join(bin_args)
            rconn = RemoteConn(remote_info["name"], remote_info["uname"], remote_info["host"], remote_info["port"], use_agent=self.remote_info.get("use_agent", False))
            local_results_fpath = None
            remote_results_fpath = None
            if "benchmark" in remote_bin_args and "benchmark_out" not in remote_bin_args:
                out_fmt = "--benchmark_out_format=json" if "benchmark_out_format" not in remote_bin_args else ""
                # Set output directory to local, that way we can reuse the conditional below
                remote_bin_args = f"{remote_bin_args} {out_fmt} --benchmark_out={out_dir}/stats.json"
            
            if "benchmark_out" in remote_bin_args:
                bench_path = None
                for sba in remote_bin_args.split():
                    if "--benchmark_out=" in sba:
                        bench_path = sba.replace("--benchmark_out=", "").strip()
                        break
                logger.info(f"{remote_bin_args.split()}")
                assert bench_path is not None, f"Unable to find output path for {remote_bin_args}"
                local_results_fpath = bench_path
                remote_results_fpath = bench_dir / "stats.json"
                remote_bin_args = remote_bin_args.replace(local_results_fpath, str(remote_results_fpath))

            remote_bin_cmd = f"{remote_bin_path} {remote_bin_args}"
            if not rconn.file_exists(f"{remote_bin_path}"):
                rconn.copy_to_remote(bin_path, remote_bin_path,  0o755)
            rconn.run_cmd(remote_bin_cmd)
            if "benchmark_out" in remote_bin_args:
                assert local_results_fpath is not None
                assert remote_results_fpath is not None
                rconn.copy_from_remote(f"{bench_dir}/stats.json", f"{out_dir}/stats.json", 0o755)
        else:
            if "benchmark" in bin_cmd and "benchmark_out" not in bin_cmd:
                logger.info(f"Storing output results to {out_dir}/stats.json")
                bin_cmd = f"{bin_cmd} --benchmark_out_format=json --benchmark_out={out_dir}/stats.json"
            sim_cmd = bin_cmd.split()
            shell(sim_cmd, cwd=str(out_dir), env=run_env)
        if self.use_qemu:
            time_ns = get_cycles(out_dir)
        else:
            time_ns = self.get_sim_results(out_dir)
        time_ns = list(time_ns.values())[0]
        if time_ns is None:
            logger.info(f"Unable to get total nanoseconds results")
        else:
            logger.info(f"Total time: {time_ns} ns")

    def sweep_sim(self, output_dir: Optional[str] = None, iters: int = 1, num_bench: int = -1, start_bench: int = 0, kernels: Optional[List[str]] = None, overwrite: bool = False, test_num: Optional[int] = None, skip_existing: bool = True):
        self.check_exists(False)
        assert "xnnpack" in self.target
 
        with open(str(BENCH_FILE), "r") as f:
            benches = [l.strip() for l in f.readlines()]
        bench_vals = {}
        for b in benches:
            s = re.search("M:(\d+)/N:(\d+)/K:(\d+)", b)
            if s is None:
                s = re.search("(\d+)/(\d+)/(\d+)/real_time", b)
            assert s is not None, f"Unable to match dimensions againt benchmark {b}"
            assert s.group(1) is not None, f"Unable to get dimension in group 1 for benchmark {b}"
            assert s.group(2) is not None, f"Unable to get dimension in group 2 for benchmark {b}"
            assert s.group(3) is not None, f"Unable to get dimension in group 3 for benchmark {b}"
            m = int(s.group(1))
            n = int(s.group(2))
            k = int(s.group(3))
            bench_vals[b] = m*n*k
        benches = sorted(benches, key=lambda b: bench_vals[b])

        # Setup results directory
        if output_dir:
            out_dir = Path(output_dir)
            out_dir = RESULTS_DIR / out_dir if not out_dir.is_absolute() else out_dir
        else:
            out_dir = RESULTS_DIR / f"sweep_{self.target}_{test_num}" if test_num is not None and test_num >= 0 else RESULTS_DIR / f"sweep_{self.target}"
        if isinstance(self.ukernels, list):
            target_kernels = []
            for kfile in self.ukernels:
                target_kernels += get_kernels(kfile)
        else:
            target_kernels = get_kernels(self.ukernels)

        if kernels:
            assert all([k in target_kernels for k in kernels])
            tgt_kernels = kernels
        else:
            tgt_kernels = target_kernels
        # Setup output directories
        out_dir.mkdir(parents=True, exist_ok=overwrite)
        set_log_file(f"{out_dir}/info.log")

        default_args = []
        default_args.append(f"--benchmark_repetitions={iters}")
        default_args.append(f"--benchmark_min_time=1x")
        default_args += ["--benchmark_min_warmup_time=0", "--benchmark_dry_run=false", "--num_threads=1"]
        num_bench = num_bench if num_bench > 0 else len(benches)
        measured = 0
        idx = start_bench
        headers = ["Benchmark", "Kernel", "Cycles", "TimeNS", "Freq"]
        all_results = []
        while idx < len(benches) and measured < num_bench:
            b = benches[idx]
            bench_name = "_".join(b.split("/")[1:]).replace(":", "").replace("_real_time", "")
            logger.info(f"Running benchmark {bench_name}: {idx - start_bench} / {num_bench}")
            bench_dir = (out_dir / bench_name)
            bench_dir.mkdir(exist_ok=overwrite)
            if self.use_qemu:
                for k in tgt_kernels:
                    filter_name = b.replace("$KERNEL_FILTER$", k)
                    out_subdir = (bench_dir / k)
                    if out_subdir.exists() and overwrite and (out_subdir / "stats" / "pilos_combined.stats").exists() and skip_existing:
                        logger.info(f"Skipping execution for {bench_name} with kernel {k} because results already exist in {out_subdir}")
                    else:
                        out_subdir.mkdir(exist_ok=overwrite)
                        logger.info(f"Storing results for {bench_name} with kernel {k} in {out_subdir}")
                        set_log_file(f"{out_subdir}/sweepsim.log")
                        in_args = " ".join((default_args + [f"--benchmark_filter={filter_name}"]))
                        self.simulate(sweep_bin, 
                            bin_args=[in_args],
                            output_dir=out_subdir)
                    perf_info = get_cycles(out_subdir, return_perf_info=True)
                    perf_info = list(perf_info.values())[0]
                    row = [filter_name.replace("/", "_").replace(":", ""), k, perf_info["cycles"], perf_info["time_ns"], perf_info["freq"]]
                    all_results.append(row)
            else:
                if bench_dir.exists() and overwrite and (bench_dir / "stats.json").exists() and skip_existing:
                    logger.info(f"Skipping execution for {bench_name} because results already exist in {bench_dir}")
                else:
                    set_log_file(f"{bench_dir}/sweepsim.log")
                    logger.info(f"Storing accumulated results for {bench_name} for all kernels in {bench_dir}")
                    kernel_filter = "(" + "|".join(tgt_kernels) + ")"
                    filter_name = b.replace("$KERNEL_FILTER$", kernel_filter)
                    if self.remote_info:
                        in_args = " ".join((default_args + [f"--benchmark_filter='{filter_name}'"]))
                    else:
                        in_args = " ".join((default_args + [f"--benchmark_filter={filter_name}"]))
                    self.simulate(sweep_bin, 
                            bin_args=[in_args],
                            output_dir=bench_dir)
                perf_info = self.get_sim_results(out_dir, return_perf_info=True)
                for pi_name, pi in perf_info.items():
                    b_row = b.replace("$KERNEL_FILTER$", pi_name).replace("/", "_").replace(":", "")
                    row = [b_row, pi_name, pi["cycles"], pi["time_ns"], pi["freq"]]
                    all_results.append(row)
            set_log_file(f"{out_dir}/info.log")
            idx += 1
        summary_path = out_dir / "sweep_summary.csv"
        with open(str(summary_path), "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows([headers] + all_results)

class SubRunner(BenchSimRunner):
    def __init__(self, root: Path):
        self._parent_root = root
    
    @property
    def parent_root(self) -> Path:
        return self._parent_root
    
    @property
    def sub_root(self) -> str:
        raise NotImplementedError("root not defined")
    
    @property
    def root(self) -> Path:
        return self._parent_root / self.sub_root

class CompoundRunner(BaseRunner):
    def __init__(self, root: Path, sub_runners: List[SubRunner]):
        assert isinstance(root, Path)
        assert all([isinstance(sr, SubRunner) and sr.parent_root == root for sr in sub_runners])
        self._sub_runners = sub_runners
        self._root = root
    
    def check_exists(self, init_on_fail: bool = True):
        assert isinstance(self.root, Path), f"Root value should be Path, not {type(self.root)}"
        if not self.root.exists() and init_on_fail:
            logger.info(f"Code for {self.target} does not yet exist locally. Proceeding to setup")
            self.init()
        elif not self.root.exists():
            logger.info(f"Code for {self.target} does not exist locally. Please initialize first")
        return self.root.exists()

    @property
    def root(self) -> str:
        return self._root

    @property
    def sub_runners(self) -> SubRunner:
        return self._sub_runners

    def pull(self, **kwargs):
        assert isinstance(self.sub_runners, list)
        self.check_exists()
        for sr in self.sub_runners:
            sr.pull(**kwargs)

    def checkout(self, **kwargs):
        for sr in self.sub_runners:
            sr.checkout(**kwargs)

    def sync(self):
        for sr in self.sub_runners:
            sr.sync(**kwargs)

    def init(self, reinit: bool = False, extras: bool = False):
        if reinit:
            shutil.rmtree(self.root_str)
        self.check_deps(False)
        if not self.root.exists():
            self.root.mkdir()

        for sr in self.sub_runners:
            sr.init(reinit, extras)
        
        env = dict(BASE_ENV, **self.env)
        for pi in self.post_init:
            shell(pi.split(), env=env, cwd=self.root_str)

    def run_commands(self, cmds: List[str], shell_cmds: bool = False):
        env = dict(BASE_ENV, **self.env)
        assert isinstance(cmds, list)
        for cmd in cmds:
            if shell_cmds:
                shell(cmd, shell=True, env=env, cwd=self.root_str)
            else:
                shell(cmd.split(), env=env, cwd=self.root_str)

    def build(self, **kwargs):
        self.check_exists()
        # Compound runner commands always come before the sub runners
        build_cmd = self.get_build_cmd(**kwargs)
        if len(build_cmd) > 0:
            self.run_commands(build_cmd, self.shell_build)
        for sr in self.sub_runners:
            sr.build(**kwargs)

    def rebuild(self, **kwargs):
        self.check_exists(False)
        # Compound runner commands always come before the sub runners
        if len(self.rebuild_cmd) > 0:
            self.run_commands(self.rebuild_cmd)
            
        for sr in self.sub_runners:
            sr.rebuild(**kwargs)

    def run(self, **kwargs):
        self.check_exists(False)
        for sr in self.sub_runners:
            sr.run(**kwargs)

    def simulate(self, **kwargs):
        self.check_exists(False)
        for sr in self.sub_runners:
            sr.simulate(**kwargs)

    def sweep_sim(self, **kwargs):
        self.check_exists(False)
        for sr in self.sub_runners:
            sr.sweep_sim(**kwargs)

class MultiRunner(BaseRunner):
    def __init__(self, target: str, subtargets: List[BaseRunner]):
        self._target = target
        assert all([isinstance(st, BaseRunner) for st in subtargets]), f"All subtargets for MultiRunner must be BaseRunner instances"
        self._subtargets = subtargets
    
    @property
    def target(self) -> str:
        return self._target

    @property
    def subtargets(self) -> List[BaseRunner]:
        return self._subtargets

    def pull(self, **kwargs):
        for st in self.subtargets:
            st.pull(**kwargs)

    def sync(self, **kwargs):
        for st in self.subtargets:
            st.sync(**kwargs)

    def checkout(self, **kwargs):
        for st in self.subtargets:
            st.checkout(**kwargs)

    def init(self, **kwargs):
        self.check_deps(True)

        for st in self.subtargets:
            st.init(**kwargs)
        env = dict(BASE_ENV, **self.env)
        for pi in self.post_init:
            shell(pi.split(), env=env, cwd=self.root_str)

    def build(self, **kwargs):
        for st in self.subtargets:
            st.build(**kwargs)

    def rebuild(self, **kwargs):
        for st in self.subtargets:
            st.rebuild(**kwargs)

    def run(self, **kwargs):
        for st in self.subtargets:
            st.run(**kwargs)

    def simulate(self, **kwargs):
        for st in self.subtargets:
            st.simulate(**kwargs)

    def sweep_sim(self, **kwargs):
        for st in self.subtargets:
            st.sweep_sim(**kwargs)

    
