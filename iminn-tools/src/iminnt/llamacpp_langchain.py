"""
LangChain integration for LlamaCppService.

This module provides LangChain-compatible LLM wrapper for RISC-V llama.cpp
inference via QEMU. It allows using extended RISC-V CPU with LangChain
frameworks to verify real-world LLM application support.

Phase 3: LangChain integration for application framework verification.
"""

from typing import Optional, List, Dict, Any
from pydantic import Field

try:
    # LangChain 1.x (new structure) - use BaseLLM
    from langchain_core.language_models.llms import BaseLLM
    from langchain_core.callbacks.manager import CallbackManagerForLLMRun
    from langchain_core.outputs import LLMResult, Generation
    LLM_BASE_CLASS = BaseLLM
    USE_GENERATE = True
except ImportError:
    # Fallback for older versions
    try:
        from langchain.llms.base import LLM
        from langchain.callbacks.manager import CallbackManagerForLLMRun
        LLM_BASE_CLASS = LLM
        USE_GENERATE = False
    except ImportError:
        raise ImportError(
            "LangChain not installed. Install with: pip install langchain langchain-core"
        )

from .llamacpp_service import LlamaCppService
from .log_cfg import logger


class RISCVRISCLLM(LLM_BASE_CLASS):
    """
    LangChain LLM wrapper for RISC-V llama.cpp via QEMU.
    
    This class wraps LlamaCppService to provide a LangChain-compatible
    interface. It allows LangChain applications to use RISC-V compiled
    llama.cpp inference, enabling verification that extended RISC-V CPUs
    can support real-world LLM application frameworks.
    
    Example:
        >>> from iminnt.llamacpp_langchain import RISCVRISCLLM
        >>> from langchain.chains import LLMChain
        >>> from langchain.prompts import PromptTemplate
        >>> 
        >>> llm = RISCVRISCLLM()
        >>> prompt = PromptTemplate(
        ...     input_variables=["topic"],
        ...     template="Write a short story about {topic}:"
        ... )
        >>> chain = LLMChain(llm=llm, prompt=prompt)
        >>> result = chain.run("a robot learning to paint")
    """
    
    # Configuration fields (Pydantic)
    llama_cli_path: Optional[str] = Field(
        default=None,
        description="Path to llama-cli binary (uses default if None)"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to GGUF model file (uses default if None)"
    )
    max_tokens: int = Field(
        default=128,
        description="Maximum number of tokens to generate",
        ge=1,
        le=4096
    )
    threads: int = Field(
        default=1,
        description="Number of threads to use",
        ge=1,
        le=4
    )
    temperature: float = Field(
        default=0.8,
        description="Sampling temperature (0.0-2.0)",
        ge=0.0,
        le=2.0
    )
    seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility",
        ge=0
    )
    extra_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional llama-cli arguments"
    )
    
    # Internal service instance (not a Pydantic field)
    _service: Optional[LlamaCppService] = None
    
    class Config:
        """Pydantic config."""
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """Initialize RISCVRISCLLM with optional configuration."""
        super().__init__(**kwargs)
        
        # Initialize LlamaCppService
        service_kwargs = {}
        if self.llama_cli_path:
            from pathlib import Path
            service_kwargs["llama_cli_path"] = Path(self.llama_cli_path)
        if self.model_path:
            from pathlib import Path
            service_kwargs["model_path"] = Path(self.model_path)
        
        try:
            self._service = LlamaCppService(**service_kwargs)
            logger.info(f"RISCVRISCLLM initialized with service: {self._service}")
        except Exception as e:
            logger.error(f"Failed to initialize LlamaCppService: {e}")
            raise
    
    @property
    def _llm_type(self) -> str:
        """Return LLM type identifier."""
        return "riscv_llamacpp"
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> LLMResult:
        """
        Generate text from prompts using RISC-V llama.cpp inference.
        
        This is the required method for LangChain 1.x BaseLLM.
        
        Args:
            prompts: List of input text prompts
            stop: Optional list of stop sequences
            run_manager: Optional callback manager for LLM run
            **kwargs: Additional generation parameters
        
        Returns:
            LLMResult containing generated text
        """
        if self._service is None:
            raise RuntimeError(
                "LlamaCppService not initialized. "
                "Check initialization errors."
            )
        
        # Merge kwargs with instance parameters
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        threads = kwargs.get("threads", self.threads)
        temperature = kwargs.get("temperature", self.temperature)
        seed = kwargs.get("seed", self.seed)
        
        # Merge extra_args
        extra_args = self.extra_args.copy() if self.extra_args else {}
        if "extra_args" in kwargs:
            extra_args.update(kwargs.pop("extra_args", {}))
        
        # Handle stop sequences
        if stop:
            logger.warning(
                "Stop sequences: will be applied post-generation"
            )
        
        generations = []
        
        try:
            # Process each prompt
            for prompt in prompts:
                # Generate text using the service
                response = self._service.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    threads=threads,
                    temperature=temperature,
                    seed=seed,
                    **extra_args
                )
                
                # Apply stop sequences manually if provided
                if stop:
                    for stop_seq in stop:
                        if stop_seq in response:
                            response = response.split(stop_seq)[0]
                
                # Create Generation object
                generation = Generation(text=response)
                generations.append([generation])
            
            # Create LLMResult
            llm_result = LLMResult(generations=generations)
            
            # Call run manager callbacks if provided
            if run_manager:
                for prompt, gen_list in zip(prompts, generations):
                    run_manager.on_llm_end(LLMResult(generations=[gen_list]))
            
            return llm_result
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}", exc_info=True)
            if run_manager:
                run_manager.on_llm_error(e)
            raise
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> str:
        """
        Generate text from a single prompt (convenience method).
        
        This calls _generate internally for compatibility.
        
        Args:
            prompt: Input text prompt
            stop: Optional list of stop sequences
            run_manager: Optional callback manager for LLM run
            **kwargs: Additional generation parameters
        
        Returns:
            Generated text response
        """
        result = self._generate(
            prompts=[prompt],
            stop=stop,
            run_manager=run_manager,
            **kwargs
        )
        # Extract text from first generation
        return result.generations[0][0].text
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return identifying parameters for this LLM."""
        params = {
            "llm_type": self._llm_type,
            "max_tokens": self.max_tokens,
            "threads": self.threads,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        
        if self._service:
            params.update({
                "llama_cli_path": str(self._service.llama_cli_path),
                "model_path": str(self._service.model_path),
                "qemu_mode": "user" if self._service.use_qemu_user else "system",
            })
        
        return params
