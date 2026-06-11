"""
LlamaCppService - Service wrapper for llama-cli execution via QEMU.

This module provides a Python service interface for running llama.cpp inference
via QEMU user mode (RISC-V emulation). It's designed for Option A (Host-Based
Orchestration) where Python frameworks run on the host and orchestrate QEMU
execution of the RISC-V compiled llama-cli binary.

Phase 1: Basic service wrapper with QEMU user mode support.
Future: Extend to QEMU system mode and add streaming support.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import subprocess
import tempfile
import re
from .constants import (
    DEV_ENV_ROOT,
    QEMU_USER_BIN,
    IMI_CPU_ALIAS,
    IMI_ENV,
    PROMPTS_DIR,
    RESULTS_DIR,
)
from .log_cfg import logger


class LlamaCppService:
    """
    Service wrapper for llama-cli execution via QEMU user mode.
    
    This class provides a simple Python interface to run llama.cpp inference
    by executing the RISC-V compiled llama-cli binary through QEMU user mode
    emulation.
    
    Example:
        >>> service = LlamaCppService()
        >>> response = service.generate("Hello, how are you?", max_tokens=32)
        >>> print(response)
    """
    
    def __init__(
        self,
        llama_cli_path: Optional[Path] = None,
        model_path: Optional[Path] = None,
        use_qemu_user: bool = True,
    ):
        """
        Initialize the LlamaCppService.
        
        Args:
            llama_cli_path: Path to llama-cli binary. If None, uses default:
                dev_env/llama.cpp/llamacpp-imi-install/bin/llama-cli
            model_path: Path to GGUF model file. If None, uses default:
                dev_env/llama.cpp/models/stories15M-q4_0.gguf
            use_qemu_user: If True, use QEMU user mode (default). If False,
                use QEMU system mode (not yet implemented).
        
        Raises:
            FileNotFoundError: If llama-cli binary or model file doesn't exist.
            ValueError: If paths are invalid.
        """
        # Set default paths
        if llama_cli_path is None:
            llama_cli_path = (
                DEV_ENV_ROOT / "llama.cpp" / "llamacpp-imi-install" / "bin" / "llama-cli"
            )
        if model_path is None:
            model_path = (
                DEV_ENV_ROOT / "llama.cpp" / "models" / "stories15M-q4_0.gguf"
            )
        
        # Convert to Path objects if strings
        self.llama_cli_path = Path(llama_cli_path)
        self.model_path = Path(model_path)
        self.use_qemu_user = use_qemu_user
        
        # Validate paths
        if not self.llama_cli_path.exists():
            raise FileNotFoundError(
                f"llama-cli binary not found: {self.llama_cli_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}"
            )
        if not QEMU_USER_BIN.exists():
            raise FileNotFoundError(
                f"QEMU user binary not found: {QEMU_USER_BIN}"
            )
        
        # Check if binary is static (affects QEMU command)
        self._is_static = self._is_static_binary(self.llama_cli_path)
        
        logger.info(f"LlamaCppService initialized:")
        logger.info(f"  llama-cli: {self.llama_cli_path}")
        logger.info(f"  model: {self.model_path}")
        logger.info(f"  QEMU mode: {'user' if use_qemu_user else 'system'}")
        logger.info(f"  binary type: {'static' if self._is_static else 'dynamic'}")
    
    @staticmethod
    def _is_static_binary(bin_path: Path) -> bool:
        """
        Check if binary is statically linked.
        
        Args:
            bin_path: Path to binary file.
        
        Returns:
            True if binary is static, False otherwise.
        """
        try:
            result = subprocess.run(
                ["file", str(bin_path)],
                capture_output=True,
                text=True,
                check=True
            )
            return "statically linked" in result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            # If 'file' command fails, assume dynamic (safer default)
            return False
    
    def _build_qemu_command(
        self,
        prompt_file: Path,
        max_tokens: int = 128,
        threads: int = 1,
        temperature: float = 0.8,
        seed: int = 42,
        **kwargs
    ) -> List[str]:
        """
        Build QEMU user mode command for executing llama-cli.
        
        Args:
            prompt_file: Path to file containing the prompt.
            max_tokens: Maximum number of tokens to generate.
            threads: Number of threads to use.
            temperature: Sampling temperature (0.0-2.0).
            seed: Random seed for reproducibility.
            **kwargs: Additional llama-cli arguments.
        
        Returns:
            List of command arguments for subprocess.
        """
        # Base QEMU command
        qemu_cmd = [
            str(QEMU_USER_BIN),
            "-cpu", IMI_CPU_ALIAS,
        ]
        
        # Add sysroot if binary is dynamic
        if not self._is_static:
            if "CROSS_SYSROOT" not in IMI_ENV:
                raise ValueError(
                    "CROSS_SYSROOT not found in IMI_ENV. "
                    "Required for dynamic binaries."
                )
            qemu_cmd.extend(["-L", str(IMI_ENV["CROSS_SYSROOT"])])
        
        # Add llama-cli binary and arguments
        llama_args = [
            str(self.llama_cli_path),
            "-m", str(self.model_path),
            "-t", str(threads),
            "-n", str(max_tokens),
            "--file", str(prompt_file),
            "--seed", str(seed),
            "-ngl", "0",  # CPU only (no GPU layers)
            "--no-warmup",  # Skip warmup iterations
            "-no-cnv",  # No conversation mode
            "-st",  # Simple tokenization
        ]
        
        # Add temperature if specified (llama-cli uses --temp)
        if temperature != 0.8:
            llama_args.extend(["--temp", str(temperature)])
        
        # Add any additional kwargs as llama-cli arguments
        for key, value in kwargs.items():
            if key.startswith("--"):
                # Already formatted as --arg
                llama_args.append(key)
                if value is not None and value != "":
                    llama_args.append(str(value))
            elif key.startswith("-"):
                # Short form -arg
                llama_args.append(key)
                if value is not None and value != "":
                    llama_args.append(str(value))
            else:
                # Convert to --arg-name format
                arg_name = f"--{key.replace('_', '-')}"
                llama_args.append(arg_name)
                if value is not None and value != "":
                    llama_args.append(str(value))
        
        qemu_cmd.extend(llama_args)
        
        return qemu_cmd
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        threads: int = 1,
        temperature: float = 0.8,
        seed: int = 42,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt using llama-cli via QEMU.
        
        Args:
            prompt: Input text prompt.
            max_tokens: Maximum number of tokens to generate (default: 128).
            threads: Number of threads to use (default: 1).
            temperature: Sampling temperature, 0.0-2.0 (default: 0.8).
            seed: Random seed for reproducibility (default: 42).
            **kwargs: Additional llama-cli arguments (e.g., top_p, top_k).
        
        Returns:
            Generated text output from llama-cli.
        
        Raises:
            subprocess.CalledProcessError: If llama-cli execution fails.
            ValueError: If prompt is empty or parameters are invalid.
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        
        if threads < 1:
            raise ValueError("threads must be >= 1")
        
        if not (0.0 <= temperature <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        
        # Write prompt to temporary file
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False
        ) as prompt_file:
            prompt_file.write(prompt)
            prompt_file_path = Path(prompt_file.name)
        
        try:
            # Build QEMU command
            cmd = self._build_qemu_command(
                prompt_file_path,
                max_tokens=max_tokens,
                threads=threads,
                temperature=temperature,
                seed=seed,
                **kwargs
            )
            
            logger.info(f"Executing llama-cli via QEMU user mode...")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            # Execute command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=IMI_ENV.copy() if IMI_ENV else None,
            )
            
            # Parse output
            # llama-cli outputs the generated text to stdout
            # We need to extract just the generated text (skip metadata/logs)
            output = self._parse_output(result.stdout, result.stderr)
            
            return output
            
        except subprocess.CalledProcessError as e:
            logger.error(f"llama-cli execution failed with exit code {e.returncode}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise
        finally:
            # Clean up temporary prompt file
            try:
                prompt_file_path.unlink()
            except Exception:
                pass
    
    def _parse_output(self, stdout: str, stderr: str) -> str:
        """
        Parse llama-cli output to extract generated text.
        
        llama-cli outputs various metadata and logs. This method extracts
        the actual generated text from the output.
        
        Args:
            stdout: Standard output from llama-cli.
            stderr: Standard error from llama-cli.
        
        Returns:
            Extracted generated text.
        """
        # Combine stdout and stderr (llama-cli may output to either)
        full_output = stdout + "\n" + stderr if stderr else stdout
        
        # llama-cli output structure:
        # 1. Warnings (warning: ...)
        # 2. Build info (build: ...)
        # 3. Model loading messages (llama_model_loader:, print_info:, load:, etc.)
        # 4. Context initialization (llama_context:, llama_kv_cache:, etc.)
        # 5. Thread initialization (main: llama threadpool init)
        # 6. Sampler configuration (sampler seed:, sampler params:)
        # 7. Generation start (generate: ...)
        # 8. **THE ACTUAL GENERATED TEXT** (appears here)
        # 9. Performance metrics (common_perf_print:, llama_memory_breakdown_print:)
        
        lines = full_output.split("\n")
        
        # Patterns that indicate metadata/logs (should be skipped)
        metadata_patterns = [
            r"^warning:",
            r"^build:",
            r"^main:",
            r"^llama_model_loader:",
            r"^llama_context_params:",
            r"^system_info:",
            r"^sampling:",
            r"^llama_print_timings:",
            r"^llama_load_model_from_file:",
            r"^llama_new_context_with_model:",
            r"^llama_kv_cache_init:",
            r"^llama_kv_cache:",
            r"^llama_context:",
            r"^llama_batch_init:",
            r"^llama_decode:",
            r"^print_info:",
            r"^load:",
            r"^load_tensors:",
            r"^common_init_from_params:",
            r"^sampler seed:",
            r"^sampler params:",
            r"^sampler chain:",
            r"^\s+repeat_last_n",  # Sampler param lines (indented)
            r"^\s+dry_multiplier",
            r"^\s+top_k",
            r"^\s+mirostat",
            r"^generate:",
            r"^common_perf_print:",
            r"^llama_memory_breakdown_print:",
            r"^eval time:",
            r"^tokens/second:",
            r"^prompt eval time:",
            r"^sample time:",
            r"^load time:",
            r"^total time:",
            r"^llama_save_session_file:",
            r"^llama_load_session_file:",
            r"^\s*\|",  # Table lines (memory breakdown)
            r"^\.+$",  # Dots line (loading indicator)
        ]
        
        # Patterns that indicate the start of performance metrics (stop collecting here)
        perf_start_patterns = [
            r"^common_perf_print:",
            r"^llama_memory_breakdown_print:",
        ]
        
        generated_lines = []
        in_generated_section = False
        perf_section_started = False
        
        for line in lines:
            # Check if we've hit the performance metrics section
            if any(re.match(pattern, line) for pattern in perf_start_patterns):
                perf_section_started = True
                break
            
            # Skip if we're in the performance section
            if perf_section_started:
                continue
            
            # Check if this line matches metadata patterns
            is_metadata = any(re.match(pattern, line) for pattern in metadata_patterns)
            
            if is_metadata:
                # We're still in metadata section, but might be close to generated text
                # Look for markers that indicate generation is about to start
                if re.match(r"^sampler seed:", line) or re.match(r"^generate:", line):
                    in_generated_section = True
                continue
            
            # If we have a non-empty line that's not metadata, it's likely generated text
            # But also check it's not just dots (loading indicator)
            if line.strip() and not re.match(r"^\.+$", line.strip()):
                generated_lines.append(line)
                in_generated_section = True
        
        # Join the generated lines
        generated_text = "\n".join(generated_lines).strip()
        
        # If we didn't find anything, try a fallback: look for text after "generate:"
        if not generated_text:
            # Find the line with "generate:" and take everything after it until performance metrics
            generate_idx = -1
            for i, line in enumerate(lines):
                if re.match(r"^generate:", line):
                    generate_idx = i
                    break
            
            if generate_idx >= 0:
                # Collect lines after "generate:" until we hit performance metrics
                for i in range(generate_idx + 1, len(lines)):
                    line = lines[i]
                    if any(re.match(pattern, line) for pattern in perf_start_patterns):
                        break
                    if line.strip() and not any(
                        re.match(pattern, line) for pattern in metadata_patterns
                    ):
                        generated_lines.append(line)
                
                generated_text = "\n".join(generated_lines).strip()
        
        # Final fallback: return full output if we still have nothing
        if not generated_text:
            logger.warning(
                "Could not parse generated text from output. "
                "Returning full output."
            )
            return full_output.strip()
        
        return generated_text
