import subprocess
from pathlib import Path
import re
import json
import signal
import sys
from .constants import CPU_FREQ, XNNPACK_RESOURCES
from .log_cfg import logger

# Utility functions
def shell(cmd, silent=False, no_fail=False, dry_run=False, **kwargs):
    assert isinstance(cmd, list) or kwargs.get("shell", False), f"Expected list but got {type(cmd)}"
    out_str = ""

    full_cmd = " ".join(cmd) if isinstance(cmd, list) and not shell else cmd
    if not silent:
        cmd_str = full_cmd if isinstance(full_cmd, str) else " ".join(full_cmd)
        logger.info(f"Running bash command: {cmd_str}")

    if isinstance(cmd, str):
        cmd = f"stdbuf -oL -eL {cmd}"
        assert kwargs.get("shell", False)
    else:
        cmd = ['stdbuf', '-oL', '-eL'] + cmd

    if dry_run:
        return
    
    # Check if this is a QEMU system mode command (for special signal handling)
    is_qemu_system = isinstance(cmd, list) and any('qemu-system' in str(arg) for arg in cmd)
    
    # Use a list to hold process reference (for closure in signal handler)
    process_ref = [None]
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C to properly terminate QEMU"""
        proc = process_ref[0]
        if proc is not None:
            logger.info("\n[Interrupted] Terminating QEMU process...")
            if is_qemu_system:
                # For QEMU system mode, try graceful shutdown first
                try:
                    # Send SIGTERM for graceful shutdown
                    proc.terminate()
                    # Wait a bit for graceful shutdown
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        # Force kill if it doesn't respond
                        logger.info("QEMU did not respond to SIGTERM, forcing termination...")
                        proc.kill()
                except Exception as e:
                    logger.warning(f"Error terminating process: {e}")
                    try:
                        proc.kill()
                    except:
                        pass
            else:
                # For other processes, just terminate
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                except:
                    pass
            sys.exit(130)  # Exit code 130 for SIGINT
    
    # Register signal handler for Ctrl+C
    old_handler = signal.signal(signal.SIGINT, signal_handler)
    
    try:
        with subprocess.Popen(
            cmd,
            **kwargs,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, 
            universal_newlines=True,
            errors='ignore',
            preexec_fn=None if sys.platform == 'win32' else lambda: signal.signal(signal.SIGINT, signal.SIG_DFL)
        ) as process:
            process_ref[0] = process  # Store reference for signal handler
            if not silent: logger.info(f"PID: {process.pid}")
            if is_qemu_system and not silent:
                logger.info("Note: Press Ctrl+C to terminate QEMU (may take a moment)")
            
            for line in iter(process.stdout.readline, ''):
                out_str = out_str + line
                if not silent:
                    logger.info(line.replace("\n", ""))
            exit_code = None
            while exit_code is None:
                exit_code = process.poll()

            assert exit_code == 0 or no_fail, "The process returned an non-zero exit code {}! (CMD: `{}`).\nError:{}".format(
                exit_code, " ".join(list(map(str, cmd))), out_str
            )
    except KeyboardInterrupt:
        # This should be handled by signal_handler, but just in case
        proc = process_ref[0]
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    finally:
        # Restore original signal handler
        signal.signal(signal.SIGINT, old_handler)
        process_ref[0] = None
    
    return {"exit_code": exit_code, "msg": out_str}


def get_perf_in_ns(cycles: int, cpu_freq = CPU_FREQ):
    return cycles / (cpu_freq / (10**9))

def get_cycles(sim_dir: Path, return_perf_info = False):
    assert sim_dir.exists(), f"Cannot get cycles for non-existant output directory: {sim_dir}"
    stat_file = (sim_dir / "stats" / "pilos_combined.stats")
    if not stat_file.exists():
        return None
    with open(str(stat_file), "r") as f:
        stat_lines = [l.strip() for l in f.readlines()]
    
    for l in stat_lines:
        s = re.search("total_cycles=(\d+)", l)
        if s:
            cycles = int(s.group(1))
            time_ns = get_perf_in_ns(float(cycles))
            if return_perf_info:
                return {0: {"cycles": cycles, "time_ns": time_ns, "freq": (cpu_freq / (10**9))}}
            else:
                return {0: time_ns}