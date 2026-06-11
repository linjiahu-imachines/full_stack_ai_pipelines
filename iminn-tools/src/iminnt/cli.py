import argparse
import os
from pathlib import Path
from typing import Dict, Optional, List
import csv
import re
# Order of imports is important here: import registry initially, followed by the import of each class
from .registry import get_class_names, get_class_by_name
from .xnnpack import XNNPACKIMI, XNNPACKX86, XNNPACKAMX, XNNPACKNeoverse
from .llamacpp import LlamaCppX86, LlamaCppIMI
from .infra import Qemu, Arctic, Pilos, Permafrost, Spike
from .rvv_tests import RISCVTest
from .litert import LiteRTX86
from .timvx import TIMVXRiscv, TIMVXX86
from .npu import NPUEnv
from .playground import PlaygroundX86, PlaygroundRISCV
from .onnx_rt import ONNXRTX86, ONNXRTIMI
from .iree import IREEIMI, IREEX86


def main():
    parser = argparse.ArgumentParser(description="Build and evaluate IMI extension benchmarks.")
    parser.add_argument("-t", "--target", choices=get_class_names(), help="Target package/binary to use.")

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # initialize dependency
    init_parser = subparsers.add_parser("init", description="Initialize source code.")
    init_parser.add_argument("-r", "--reinit", action="store_true", help="Reinitialize by deleting the previous source code and reinitializing.")
    init_parser.add_argument("-e", "--extras", action="store_true", help="Only update additional scripts.")

    # pull latest
    pull_parser = subparsers.add_parser("pull", description="Pull latest a source code.")

    # List default run/sim options
    default_parser = subparsers.add_parser("defaults", description="List default run/sim names and commands.")
    default_parser.add_argument("-n", "--names-only", action="store_true", help="Only print the default names instead of names and binary commands.")

    # checkout branch
    checkout_parser = subparsers.add_parser("checkout", description="Checkout a specific branch.")
    checkout_parser.add_argument("-b", "--branch", required=True, help="Branch name to checkout.")

    # sync upstream
    sync_parser = subparsers.add_parser("sync", description="For valid targets, sync code with upstream to the upstream branch.")

    # build
    build_parser = subparsers.add_parser("build", description="Clean and recompile a project/piece of code.")
    build_parser.add_argument("--threads", type=int, default=None, required=False, help="Number of threads to use when building")


    # rebuild
    rebuild_parser = subparsers.add_parser("rebuild", description="Recompile a binary.")

    # run
    run_parser = subparsers.add_parser("run", description="Run an executable via qemu.")
    run_parser.add_argument("-b", "--bin", default=None, required=False, help="Binary to use for execution, if different than the default")
    run_parser.add_argument("-a", "--bin-args", nargs=argparse.REMAINDER, default=[], help="Arguments to pass to the executable.")
    run_parser.add_argument("-d", "--default-cmd", default=None, required=False, help="Run one of several different default commands from a given target.")
    run_parser.add_argument("-i", "--icount", action="store_true", help="Collect the instruction count when running the program.")
    run_parser.add_argument("--system-mode", action="store_true", help="Use QEMU system mode instead of user mode (enables multi-core support)")
    run_parser.add_argument("--smp", type=int, default=4, help="Number of CPU cores for system mode (default: 4)")
    run_parser.add_argument("--kernel", type=str, default=None, help="Path to kernel image for system mode (overrides default)")
    run_parser.add_argument("--rootfs", type=str, default=None, help="Path to rootfs image for system mode (overrides default)")

    # simulate
    sim_parser = subparsers.add_parser("sim", description="Simulate an executable via qemu/permafrost.")
    sim_parser.add_argument("-b", "--bin", default=None, required=False, help="Binary to use for execution, if different than the default")
    sim_parser.add_argument("-a", "--bin-args", nargs=argparse.REMAINDER, default=[], help="Arguments to pass to the executable.")
    sim_parser.add_argument("-d", "--default-cmd", default=None, required=False, help="Run one of several different default commands from a given target.")
    sim_parser.add_argument("-o", "--output-dir", default=None, required=False, help="Specify output directory for results.")
    sim_parser.add_argument("-p", "--print-cfg", action="store_true", help="Print configuration.")
    sim_parser.add_argument("-c", "--custom-cfg", default=None, required=False, help="Specify custom config to use for simulation.")
    sim_parser.add_argument("-k", "--keep-retires", action="store_true", help="Preserve retired instructions for debugging purposes.")
    sim_parser.add_argument("-n", "--no-roi", action="store_true", help="Profile the full program, using no roi.")
    sim_parser.add_argument("-g", "--fn-perf-graph", action="store_true", help="Generate a trace file representing a graph of function call performance.")

    # simpoint
    simpoint_parser = subparsers.add_parser("simpoint", description="Simpoint an executable via qemu/permafrost.")
    simpoint_parser.add_argument("-b", "--bin", default=None, required=False, help="Binary to use for execution, if different than the default")
    simpoint_parser.add_argument("-a", "--bin-args", nargs=argparse.REMAINDER, default=[], help="Arguments to pass to the executable.")
    simpoint_parser.add_argument("-d", "--default-cmd", default=None, required=False, help="Run one of several different default commands from a given target.")
    simpoint_parser.add_argument("-o", "--output-dir", default=None, required=False, help="Specify output directory for results.")
    simpoint_parser.add_argument("-p", "--print-cfg", action="store_true", help="Print configuration.")
    simpoint_parser.add_argument("-c", "--custom-cfg", default=None, required=False, help="Specify custom config to use for simulation.")
    simpoint_parser.add_argument("-k", "--keep-retires", action="store_true", help="Preserve retired instructions for debugging purposes.")
    simpoint_parser.add_argument("-i", "--isolate-program", action="store_true", help="Constrain simpointing to the start and end of the program (exclude initialization instructions).")
    simpoint_parser.add_argument("-g", "--fn-perf-graph", action="store_true", help="Generate a trace file representing a graph of function call performance.")

    # sweep simulation execution
    sweep_parser = subparsers.add_parser("sweep", description="Sweep benchmark execution for a given target.")
    sweep_parser.add_argument("-o", "--output-dir", default=None, required=False, help="Specify output directory for all results.")
    sweep_parser.add_argument("-t", "--test-num", type=int, default=None, required=False, help="As an alternative to specifying an output directory, this specifies a number to append to the results folder.")
    sweep_parser.add_argument("-i", "--iters", type=int, default=1, required=False, help="Specify number of iterations per benchmark.")
    sweep_parser.add_argument("-n", "--num-bench", type=int, default=-1, required=False, help="Specify number benchmarks to run.")
    sweep_parser.add_argument("-s", "--start-bench", type=int, default=0, required=False, help="Specify the benchmark to start from.")
    sweep_parser.add_argument("-k", "--kernels", nargs="+", help="Specify the kernels to run.")
    sweep_parser.add_argument("-ow", "--overwrite", action="store_true", help="When set, will overwrite existing directories.")


    args = parser.parse_args()
    dev_fw = get_class_by_name(args.target)()
    if args.subcommand == "pull":
        kwargs = {}
    elif args.subcommand == "checkout":
        kwargs = {"branch": args.branch}
    elif args.subcommand == "init":
        kwargs = {"reinit": args.reinit, "extras": args.extras}
    elif args.subcommand == "sync":
        kwargs = {}
    elif args.subcommand == "build":
        kwargs = {"threads": args.threads}
    elif args.subcommand == "rebuild":
        kwargs = {}
    elif args.subcommand == "simpoint":
        kwargs = {"bin_name": args.bin,
                    "bin_args": args.bin_args, 
                    "default_cmd": args.default_cmd, 
                    "output_dir": args.output_dir,
                    "print_cfg": args.print_cfg, 
                    "isolate_program": args.isolate_program, 
                    "keep_retires": args.keep_retires, 
                    "custom_cfg": args.custom_cfg, 
                    "fn_perf_graph": args.fn_perf_graph}
    elif args.subcommand == "defaults":
        kwargs = {"names_only": args.names_only}
    elif args.subcommand == "run":
        kwargs = {
            "bin_name": args.bin, 
            "bin_args": args.bin_args, 
            "default_cmd": args.default_cmd, 
            "icount": args.icount,
            "system_mode": args.system_mode,
            "smp_cores": args.smp,
            "kernel_path": args.kernel,
            "rootfs_path": args.rootfs
        }
    elif args.subcommand == "sim":
        kwargs = {"bin_name": args.bin, 
                    "bin_args": args.bin_args, 
                    "default_cmd": args.default_cmd, 
                    "output_dir": args.output_dir, 
                    "print_cfg": args.print_cfg, 
                    "keep_retires": args.keep_retires, 
                    "custom_cfg": args.custom_cfg, 
                    "no_roi": args.no_roi,
                    "fn_perf_graph": args.fn_perf_graph}
    elif args.subcommand == "sweep":
        kwargs = {"output_dir": args.output_dir, "iters": args.iters, "num_bench": args.num_bench, "start_bench": args.start_bench, "kernels": args.kernels, "overwrite": args.overwrite, "test_num": args.test_num}
    else:
        raise ValueError(f"Unsupported command: {args.subcommand}")
    dev_fw.exec_cmd(args.subcommand, **kwargs)
